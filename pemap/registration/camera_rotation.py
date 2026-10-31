from __future__ import annotations

import hashlib
import json
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import networkx as nx
import numpy as np

from pemap.geometry.warp import WarpContext
from pemap.geometry.warp import build_warp_context
from pemap.registration.pipeline import optimize_camera_rotation
from pemap.registration.rvecmat import _as_rmat, _as_rvec

@dataclass(frozen=True)
class AlignedViews:
    context: WarpContext
    priors: dict[str, np.ndarray]
    rotations: dict[str, np.ndarray]
    report: dict


FEATURE_EXPOSURE_PERCENTILE = 99.0
FEATURE_TARGET_PERCENTILE_VALUE = 0.95
FEATURE_GAMMA = 2.2
FEATURE_MASK_EROSION_RADIUS = 8
CAMERA_ROTATION_CACHE_VERSION = 1
DEFAULT_CAMERA_ROTATION_MATCHERS = [
    "sift-lightglue",
    "superpoint-lightglue",
    "aliked-lightglue",
    "gim-lightglue",
    "xfeat-lightglue",
]


def _disabled_cache_info() -> dict[str, Any]:
    return {"enabled": False, "status": "disabled", "path": None, "key": None}


@dataclass(frozen=True)
class CameraRotationEstimationResult:
    rvec_by_id: dict[str, np.ndarray]
    rotation_matrix_by_id: dict[str, np.ndarray]
    used_prior: bool
    failure_reason: str | None
    graph_stats: dict[str, Any]
    cache_info: dict[str, Any] = field(default_factory=_disabled_cache_info)

    @property
    def rotation_matrices_by_id(self) -> dict[str, np.ndarray]:
        return self.rotation_matrix_by_id

    @property
    def rmat_by_id(self) -> dict[str, np.ndarray]:
        return self.rotation_matrix_by_id


def bgr_luminance(image: np.ndarray) -> np.ndarray:
    """Compute Rec.709 luminance for OpenCV BGR images."""
    image = np.asarray(image, dtype=np.float32)
    return 0.0722 * image[..., 0] + 0.7152 * image[..., 1] + 0.2126 * image[..., 2]


def estimate_global_exposure(
    images: list[np.ndarray],
    masks: list[np.ndarray] | None = None,
    *,
    percentile: float = FEATURE_EXPOSURE_PERCENTILE,
    target_value: float = FEATURE_TARGET_PERCENTILE_VALUE,
) -> tuple[float, float]:
    luminance_values = []
    if masks is None:
        masks = [np.ones(image.shape[:2], dtype=bool) for image in images]
    for image, mask in zip(images, masks):
        luminance = bgr_luminance(image)
        valid = np.asarray(mask, dtype=bool) & np.isfinite(luminance) & (luminance > 0)
        if np.any(valid):
            luminance_values.append(luminance[valid])

    if not luminance_values:
        return 1.0, 0.0

    reference_luminance = float(np.percentile(np.concatenate(luminance_values), percentile))
    if not np.isfinite(reference_luminance) or reference_luminance <= 0:
        return 1.0, reference_luminance

    return float(target_value / reference_luminance), reference_luminance


def make_feature_image(
    image: np.ndarray,
    exposure: float,
    *,
    gamma: float = FEATURE_GAMMA,
) -> np.ndarray:
    image_exposed = np.asarray(image, dtype=np.float32) * float(exposure)
    image_exposed = np.nan_to_num(image_exposed, nan=0.0, posinf=0.0, neginf=0.0)
    image_exposed = np.clip(image_exposed, 0.0, 1.0)
    return (image_exposed ** (1.0 / float(gamma))).astype(np.float32, copy=False)


def make_feature_mask(
    mask: np.ndarray,
    *,
    erosion_radius: int = FEATURE_MASK_EROSION_RADIUS,
) -> np.ndarray:
    feature_mask = np.asarray(mask, dtype=bool)
    erosion_radius = int(erosion_radius)
    if erosion_radius < 0:
        raise ValueError(f"erosion_radius must be non-negative, got {erosion_radius}.")
    if erosion_radius == 0 or not np.any(feature_mask):
        return feature_mask

    kernel_size = 2 * erosion_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(
        feature_mask.astype(np.uint8),
        kernel,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)


def estimate_camera_rotation_matrices(
    img_bgr_stokes_by_id: dict[str, np.ndarray],
    rvec_by_id: dict[str, np.ndarray],
    intrinsics: dict,
    mask_dir: str | Path,
    pitches: list[int],
    matcher_name: str | list[str] | None = None,
    warp_context: WarpContext | None = None,
    feature_exposure_percentile: float = FEATURE_EXPOSURE_PERCENTILE,
    feature_target_percentile_value: float = FEATURE_TARGET_PERCENTILE_VALUE,
    feature_gamma: float = FEATURE_GAMMA,
    feature_mask_erosion_radius: int = FEATURE_MASK_EROSION_RADIUS,
    prior_rotation_gate_deg: float = 80.0,
    huber_scale_deg: float = 5.0,
    cache_dir: str | Path | None = None,
) -> CameraRotationEstimationResult:
    """Estimate camera rotations for PEnvMap views, falling back to prior rvecs on failure."""
    matcher = DEFAULT_CAMERA_ROTATION_MATCHERS if matcher_name is None else matcher_name
    cache_info = _disabled_cache_info()

    if not img_bgr_stokes_by_id:
        return _prior_result(rvec_by_id, "no input images", {"nodes": 0, "edges": 0, "num_nodes": 0, "num_edges": 0}, cache_info=cache_info)

    graph: nx.Graph | None = None
    cache_path: Path | None = None
    cache_key: str | None = None
    cache_load_failed = False
    try:
        if warp_context is None:
            first_payload = next(iter(img_bgr_stokes_by_id.values()))
            source_hw = np.asarray(first_payload).shape[:2]
            warp_context = build_warp_context(
                img_bgr_stokes_by_id=img_bgr_stokes_by_id,
                intrinsics=intrinsics,
                mask_dir=mask_dir,
                pitches=pitches,
                source_hw=(int(source_hw[0]), int(source_hw[1])),
            )

        if cache_dir is not None:
            cache_dir_path = Path(cache_dir)
            try:
                cache_key = _camera_rotation_cache_key(
                    img_bgr_stokes_by_id=img_bgr_stokes_by_id,
                    rvec_by_id=rvec_by_id,
                    warp_context=warp_context,
                    pitches=pitches,
                    matcher_name=matcher,
                    feature_exposure_percentile=feature_exposure_percentile,
                    feature_target_percentile_value=feature_target_percentile_value,
                    feature_gamma=feature_gamma,
                    feature_mask_erosion_radius=feature_mask_erosion_radius,
                    prior_rotation_gate_deg=prior_rotation_gate_deg,
                    huber_scale_deg=huber_scale_deg,
                )
                cache_path = cache_dir_path / f"{cache_key}.npz"
                cached_result, cache_info = _read_camera_rotation_cache(cache_path, cache_key)
                cache_load_failed = cache_info["status"] == "load_failed"
                if cached_result is not None:
                    cached_rvec_by_id, cached_graph_stats = cached_result
                    print(f"Camera rotation cache hit: {cache_path}")
                    return CameraRotationEstimationResult(
                        rvec_by_id=cached_rvec_by_id,
                        rotation_matrix_by_id={bracket_id: _as_rmat(rvec) for bracket_id, rvec in cached_rvec_by_id.items()},
                        used_prior=False,
                        failure_reason=None,
                        graph_stats=cached_graph_stats,
                        cache_info=cache_info,
                    )
            except Exception as exc:
                warnings.warn(f"Camera rotation cache unavailable; recomputing. Reason: {exc}", RuntimeWarning, stacklevel=2)
                cache_info = _camera_rotation_cache_info(True, "load_failed", cache_dir_path, cache_key, reason=str(exc))
                cache_path = None
                cache_key = None
                cache_load_failed = True

        graph = _build_feature_graph(
            img_bgr_stokes_by_id=img_bgr_stokes_by_id,
            rvec_by_id=rvec_by_id,
            camera_matrix=np.asarray(warp_context.camera_matrix, dtype=np.float32).reshape(3, 3),
            dist_coeffs=np.asarray(warp_context.distortion, dtype=np.float32).reshape(-1),
            warp_context=warp_context,
            feature_exposure_percentile=feature_exposure_percentile,
            feature_target_percentile_value=feature_target_percentile_value,
            feature_gamma=feature_gamma,
            feature_mask_erosion_radius=feature_mask_erosion_radius,
        )
        graph = optimize_camera_rotation(
            graph,
            matcher_name=matcher,
            prior_rotation_gate_deg=prior_rotation_gate_deg,
            huber_scale_deg=huber_scale_deg,
        )
        graph_stats = _graph_stats(graph)
        graph_stats["matcher_name"] = matcher
        if graph.number_of_edges() == 0:
            return _prior_result(rvec_by_id, "no valid rotation constraints", graph_stats, cache_info=cache_info)

        node_ids = sorted(graph.nodes())
        optimized_rvec_by_id: dict[str, np.ndarray] = {}
        for node_idx in node_ids:
            node_data = graph.nodes[node_idx]
            bracket_id = str(node_data["bracket_id"])
            if "rvec" not in node_data:
                return _prior_result(rvec_by_id, f"optimizer produced no rvec for {bracket_id}", graph_stats, cache_info=cache_info)
            optimized_rvec_by_id[bracket_id] = _as_rvec(node_data["rvec"]).reshape(3, 1)

        if cache_path is not None and cache_key is not None:
            save_info = _write_camera_rotation_cache(cache_path, cache_key, optimized_rvec_by_id, graph_stats, warp_context)
            if save_info["status"] == "save_failed":
                cache_info = save_info
            elif cache_load_failed:
                cache_info = {**cache_info, "repaired": True}
            else:
                cache_info = save_info

        return CameraRotationEstimationResult(
            rvec_by_id=optimized_rvec_by_id,
            rotation_matrix_by_id={bracket_id: _as_rmat(rvec) for bracket_id, rvec in optimized_rvec_by_id.items()},
            used_prior=False,
            failure_reason=None,
            graph_stats=graph_stats,
            cache_info=cache_info,
        )
    except Exception as exc:
        graph_stats = _graph_stats(graph) if graph is not None else {}
        graph_stats["matcher_name"] = matcher
        graph_stats["optimizer_failed"] = True
        return _prior_result(rvec_by_id, f"{type(exc).__name__}: {exc}", graph_stats, cache_info=cache_info)


def _build_feature_graph(
    *,
    img_bgr_stokes_by_id: dict[str, np.ndarray],
    rvec_by_id: dict[str, np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    warp_context: WarpContext,
    feature_exposure_percentile: float,
    feature_target_percentile_value: float,
    feature_gamma: float,
    feature_mask_erosion_radius: int,
) -> nx.Graph:
    images_for_exposure: list[np.ndarray] = []
    masks_for_exposure: list[np.ndarray] = []
    node_inputs: list[tuple[int, str, np.ndarray, np.ndarray]] = []

    for node_idx, bracket_id in enumerate(warp_context.sorted_ids):
        if bracket_id not in img_bgr_stokes_by_id:
            raise ValueError(f"Missing image payload for bracket_id={bracket_id}")
        if bracket_id not in rvec_by_id:
            raise ValueError(f"Missing prior rvec for bracket_id={bracket_id}")

        payload = np.asarray(img_bgr_stokes_by_id[bracket_id], dtype=np.float32)
        if payload.ndim != 4 or payload.shape[2] != 3 or payload.shape[3] < 1:
            raise ValueError(f"Expected BGR-Stokes payload shape (H, W, 3, S), got {payload.shape} for bracket_id={bracket_id}")

        image = payload[:, :, :, 0]
        finite_mask = np.all(np.isfinite(image), axis=2)
        mask = warp_context.masks_by_pitch[warp_context.pitch_by_id[bracket_id]] >= 128
        if mask.shape != image.shape[:2]:
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {image.shape[:2]} for bracket_id={bracket_id}")

        image_valid = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        feature_mask = mask & finite_mask
        image_undistorted, mask_undistorted = _undistort_feature_inputs(image_valid, feature_mask, camera_matrix, dist_coeffs)
        feature_mask_undistorted = make_feature_mask(mask_undistorted, erosion_radius=feature_mask_erosion_radius)

        images_for_exposure.append(image_undistorted)
        masks_for_exposure.append(mask_undistorted)
        node_inputs.append((node_idx, bracket_id, image_undistorted, feature_mask_undistorted))

    exposure, reference_luminance = estimate_global_exposure(
        images_for_exposure,
        masks_for_exposure,
        percentile=feature_exposure_percentile,
        target_value=feature_target_percentile_value,
    )
    print("Feature exposure: " f"percentile={feature_exposure_percentile:.1f}, " f"reference_luminance={reference_luminance:.6g}, " f"scale={exposure:.6g}, gamma={feature_gamma:.2f}")

    graph = nx.Graph()
    for node_idx, bracket_id, image_undistorted, mask_undistorted in node_inputs:
        graph.add_node(
            node_idx,
            bracket_id=bracket_id,
            feature_image=make_feature_image(image_undistorted, exposure, gamma=feature_gamma),
            feature_mask=mask_undistorted,
            camera_matrix=camera_matrix.astype(np.float32, copy=False),
            rvec_prior=np.asarray(rvec_by_id[bracket_id], dtype=np.float32).reshape(3, 1),
        )
    graph.graph["feature_exposure"] = exposure
    graph.graph["feature_reference_luminance"] = reference_luminance
    return graph


def _undistort_feature_inputs(
    image: np.ndarray,
    mask: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dist = np.asarray(dist_coeffs, dtype=np.float32).reshape(-1)
    if dist.size == 0 or not np.any(np.isfinite(dist) & (np.abs(dist) > 0.0)):
        return image.astype(np.float32, copy=False), np.asarray(mask, dtype=bool)

    K = np.asarray(camera_matrix, dtype=np.float32).reshape(3, 3)
    undistorted_image = cv2.undistort(image.astype(np.float32, copy=False), K, dist, None, K)
    mask_u8 = np.asarray(mask, dtype=bool).astype(np.uint8) * 255
    undistorted_mask = cv2.undistort(mask_u8, K, dist, None, K) >= 128
    return undistorted_image.astype(np.float32, copy=False), undistorted_mask


def _camera_rotation_cache_info(
    enabled: bool,
    status: str,
    path: str | Path | None,
    key: str | None,
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    info = {
        "enabled": bool(enabled),
        "status": status,
        "path": str(path) if path is not None else None,
        "key": key,
    }
    if reason is not None:
        info["reason"] = reason
    return info


def _ordered_bracket_ids(warp_context: WarpContext, img_bgr_stokes_by_id: dict[str, np.ndarray]) -> list[str]:
    context_ids = [str(bracket_id) for bracket_id in getattr(warp_context, "sorted_ids", [])]
    if context_ids and all(bracket_id in img_bgr_stokes_by_id for bracket_id in context_ids):
        return context_ids
    return sorted(str(bracket_id) for bracket_id in img_bgr_stokes_by_id)


def _normalise_matcher_for_cache(matcher_name: str | list[str] | None) -> str | list[str]:
    if matcher_name is None:
        return list(DEFAULT_CAMERA_ROTATION_MATCHERS)
    if isinstance(matcher_name, str):
        return matcher_name
    return [str(name) for name in matcher_name]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _array_digest(array: np.ndarray) -> dict[str, Any]:
    arr = np.ascontiguousarray(np.asarray(array))
    if arr.dtype == object:
        raise TypeError("Object arrays cannot be used for camera rotation cache fingerprints.")
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(arr.tobytes())
    return {"dtype": str(arr.dtype), "shape": list(arr.shape), "sha256": digest.hexdigest()}


def _image_s0_digest(payload: np.ndarray, bracket_id: str) -> dict[str, Any]:
    arr = np.asarray(payload)
    if arr.ndim != 4 or arr.shape[2] != 3 or arr.shape[3] < 1:
        raise ValueError(f"Expected BGR-Stokes payload shape (H, W, 3, S), got {arr.shape} for bracket_id={bracket_id}")
    return _array_digest(arr[:, :, :, 0])


def _camera_rotation_cache_key(
    *,
    img_bgr_stokes_by_id: dict[str, np.ndarray],
    rvec_by_id: dict[str, np.ndarray],
    warp_context: WarpContext,
    pitches: list[int],
    matcher_name: str | list[str] | None,
    feature_exposure_percentile: float,
    feature_target_percentile_value: float,
    feature_gamma: float,
    feature_mask_erosion_radius: int,
    prior_rotation_gate_deg: float,
    huber_scale_deg: float,
) -> str:
    ordered_ids = _ordered_bracket_ids(warp_context, img_bgr_stokes_by_id)
    pitch_by_id = getattr(warp_context, "pitch_by_id", {})
    masks_by_pitch = getattr(warp_context, "masks_by_pitch", {})

    payload = {
        "version": CAMERA_ROTATION_CACHE_VERSION,
        "ordered_ids": ordered_ids,
        "pitches": [int(pitch) for pitch in pitches],
        "matcher_name": _normalise_matcher_for_cache(matcher_name),
        "feature": {
            "exposure_percentile": float(feature_exposure_percentile),
            "target_percentile_value": float(feature_target_percentile_value),
            "gamma": float(feature_gamma),
            "mask_erosion_radius": int(feature_mask_erosion_radius),
        },
        "optimizer": {
            "prior_rotation_gate_deg": float(prior_rotation_gate_deg),
            "huber_scale_deg": float(huber_scale_deg),
        },
        "camera_matrix": _array_digest(np.asarray(warp_context.camera_matrix, dtype=np.float32).reshape(3, 3)),
        "distortion": _array_digest(np.asarray(warp_context.distortion, dtype=np.float32).reshape(-1)),
        "views": [
            {
                "id": bracket_id,
                "pitch": int(pitch_by_id[bracket_id]) if bracket_id in pitch_by_id else None,
                "prior_rvec": _array_digest(np.asarray(rvec_by_id[bracket_id], dtype=np.float32).reshape(3, 1)),
                "image_s0": _image_s0_digest(img_bgr_stokes_by_id[bracket_id], bracket_id),
            }
            for bracket_id in ordered_ids
        ],
        "masks": [{"pitch": int(pitch), "mask": _array_digest(np.asarray(mask))} for pitch, mask in sorted(masks_by_pitch.items(), key=lambda item: (0, int(item[0])) if isinstance(item[0], (int, np.integer)) else (1, str(item[0])))],
    }
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_camera_rotation_cache(
    cache_path: Path,
    cache_key: str,
) -> tuple[tuple[dict[str, np.ndarray], dict[str, Any]] | None, dict[str, Any]]:
    if not cache_path.exists():
        return None, _camera_rotation_cache_info(True, "miss", cache_path, cache_key)

    try:
        with np.load(cache_path, allow_pickle=False) as data:
            for name in ("metadata_json", "ids", "rvecs"):
                if name not in data:
                    raise ValueError(f"missing {name!r}")
            metadata = json.loads(str(data["metadata_json"].item()))
            if int(metadata.get("version", -1)) != CAMERA_ROTATION_CACHE_VERSION:
                raise ValueError(f"unsupported cache version {metadata.get('version')!r}")
            if metadata.get("key") != cache_key:
                raise ValueError("cache key mismatch")

            ids = [str(bracket_id) for bracket_id in np.asarray(data["ids"]).tolist()]
            rvecs = np.asarray(data["rvecs"], dtype=np.float32)
            if rvecs.shape != (len(ids), 3, 1):
                raise ValueError(f"invalid rvec shape {rvecs.shape}; expected {(len(ids), 3, 1)}")
            if len(set(ids)) != len(ids):
                raise ValueError("duplicate bracket ids")

            rvec_by_id = {bracket_id: rvecs[index].reshape(3, 1) for index, bracket_id in enumerate(ids)}
            graph_stats = metadata.get("graph_stats", {})
            return (rvec_by_id, graph_stats), _camera_rotation_cache_info(True, "hit", cache_path, cache_key)
    except Exception as exc:
        warnings.warn(f"Failed to load camera rotation cache {cache_path}: {exc}", RuntimeWarning, stacklevel=2)
        return None, _camera_rotation_cache_info(True, "load_failed", cache_path, cache_key, reason=str(exc))


def _write_camera_rotation_cache(
    cache_path: Path,
    cache_key: str,
    rvec_by_id: dict[str, np.ndarray],
    graph_stats: dict[str, Any],
    warp_context: WarpContext,
) -> dict[str, Any]:
    tmp_path: Path | None = None
    try:
        ordered_ids = _ordered_bracket_ids(warp_context, rvec_by_id)
        rvecs = np.stack([np.asarray(rvec_by_id[bracket_id], dtype=np.float32).reshape(3, 1) for bracket_id in ordered_ids], axis=0)
        metadata = {
            "version": CAMERA_ROTATION_CACHE_VERSION,
            "key": cache_key,
            "graph_stats": _jsonable(graph_stats),
        }

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=f".{cache_path.stem}.", suffix=".tmp", dir=cache_path.parent, delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            np.savez_compressed(
                tmp_file,
                metadata_json=np.array(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
                ids=np.array(ordered_ids, dtype=np.str_),
                rvecs=rvecs,
            )
        tmp_path.replace(cache_path)
        print(f"Saved camera rotation cache: {cache_path}")
        return _camera_rotation_cache_info(True, "saved", cache_path, cache_key)
    except Exception as exc:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        warnings.warn(f"Failed to save camera rotation cache {cache_path}: {exc}", RuntimeWarning, stacklevel=2)
        return _camera_rotation_cache_info(True, "save_failed", cache_path, cache_key, reason=str(exc))


def _prior_result(
    rvec_by_id: dict[str, np.ndarray],
    failure_reason: str | None,
    graph_stats: dict[str, Any],
    cache_info: dict[str, Any] | None = None,
) -> CameraRotationEstimationResult:
    prior_rvec_by_id = {str(bracket_id): np.asarray(rvec, dtype=np.float32).reshape(3, 1) for bracket_id, rvec in rvec_by_id.items()}
    return CameraRotationEstimationResult(
        rvec_by_id=prior_rvec_by_id,
        rotation_matrix_by_id={bracket_id: _as_rmat(rvec) for bracket_id, rvec in prior_rvec_by_id.items()},
        used_prior=True,
        failure_reason=failure_reason,
        graph_stats=graph_stats,
        cache_info=_disabled_cache_info() if cache_info is None else cache_info,
    )


def _graph_stats(graph: nx.Graph) -> dict[str, Any]:
    edge_weights = [float(data.get("weight", 0.0)) for _, _, data in graph.edges(data=True)]
    nodes = int(graph.number_of_nodes())
    edges = int(graph.number_of_edges())
    return {
        "nodes": nodes,
        "edges": edges,
        "num_nodes": nodes,
        "num_edges": edges,
        "edge_weight_sum": float(np.sum(edge_weights)) if edge_weights else 0.0,
        "edge_weight_mean": float(np.mean(edge_weights)) if edge_weights else 0.0,
        "feature_exposure": float(graph.graph.get("feature_exposure", 1.0)),
        "feature_reference_luminance": float(graph.graph.get("feature_reference_luminance", 0.0)),
    }


__all__ = [
    "CameraRotationEstimationResult",
    "DEFAULT_CAMERA_ROTATION_MATCHERS",
    "bgr_luminance",
    "estimate_camera_rotation_matrices",
    "estimate_global_exposure",
    "make_feature_image",
    "make_feature_mask",
]
