import numpy as np
import numpy.typing as npt


class UInt16HDRMerger:
    """The same float32 Gaussian merge with precomputed uint16 sensor weights.

    Sensor codes have only 65536 possible values. Reuse their weights across
    strips/views rather than normalizing and evaluating exp for every sample.
    """
    def __init__(self, z_min=0.05, z_max=0.95):
        if z_max <= z_min:
            raise ValueError('Require z_max > z_min')
        self.z_min, self.z_max = z_min, z_max
        self.values = np.arange(65536, dtype=np.float32)/65535.
        self.valid = (self.values >= z_min) & (self.values <= z_max)
        self.saturated = self.values >= z_max
        center, scale = .5*(z_min+z_max), .5*(z_max-z_min)
        self.weights = np.exp(-4.*((self.values-center)/scale)**2)*self.valid
        self.weighted_values = self.weights*self.values

    def merge(self, raw, times):
        raw = np.asarray(raw)
        times = np.asarray(times, np.float32)
        if raw.dtype != np.uint16 or raw.ndim < 2:
            raise ValueError('Expected a uint16 exposure stack')
        if times.ndim != 1 or len(times) != len(raw) or not np.isfinite(times).all() or np.any(times <= 0):
            raise ValueError('Invalid measured exposure times')
        weighted = self.weighted_values[raw]
        weighted *= (1./times).reshape((-1,)+(1,)*(raw.ndim-1))
        numerator = np.sum(weighted, axis=0)
        del weighted
        denominator = np.sum(self.weights[raw], axis=0)
        result = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        below = self.values[raw.max(axis=0)] < self.z_min
        above = self.values[raw.min(axis=0)] > self.z_max
        result[below] = self.z_min/np.max(times)
        result[above] = self.z_max/np.min(times)
        return result


def fuse_exposure_bracket(
    images: npt.ArrayLike,
    times: npt.ArrayLike,
    *,
    z_min: float = 0.05,
    z_max: float = 0.95,
    dtype: npt.DTypeLike = None,
) -> np.ndarray:
    """Fuse a stack of exposure-bracketed linear images into an HDR radiance map.

    Parameters
    ----------
    images : array_like
        Array of shape (N, ...). First axis indexes exposures. The images are linear intensity, not gamma encoded.
    times : array_like
        Exposure times of shape (N,). Must be positive and align with images.
    z_min, z_max : float
        Intensity window considered reliable (inclusive). Outside gets zero weight.
    dtype : dtype
        Desired data type for computation. It must be a floating point type.

    Returns
    -------
    hdr : ndarray
        HDR radiance estimate with shape images.shape[1:].

    Example
    -------
    >>> import numpy as np
    >>> import cv2
    >>> imgs = [cv2.imread(f"img_{i}.png", -1) for i in range(5)]
    >>> times = np.array([1 / 4.0, 1 / 2.0, 1.0, 2.0, 4.0])
    >>> hdr = fuse_exposure_bracket(imgs, times, z_min=1, z_max=254)
    """
    if not np.issubdtype(np.dtype(dtype), np.floating):
        raise ValueError("`dtype` must be a floating")

    img_stack = np.asarray(images, dtype=dtype)
    exp_times = np.asarray(times, dtype=dtype)
    exp_times = exp_times.reshape((-1,) + (1,) * (img_stack.ndim - 1))

    if img_stack.shape[0] != exp_times.shape[0]:
        raise ValueError("First dimension of `images` must match length of `times`")
    if np.any(exp_times <= 0):
        raise ValueError("All exposure times must be positive")
    if z_max <= z_min:
        raise ValueError("Require z_max > z_min")

    # Validity mask
    valid_low = img_stack >= z_min
    valid_high = img_stack <= z_max
    valid = valid_low & valid_high

    # Gaussian weights
    # This weighting scheme favors mid-range unsaturated pixels.
    # Pixels outside [z_min, z_max] are zero weighted.
    c = 0.5 * (z_min + z_max)
    s = 0.5 * (z_max - z_min)
    w = np.exp(-4.0 * ((img_stack - c) / s) ** 2) * valid

    # Weighted merge (irradiance estimate ~ I / t)
    inv_exp_times = 1.0 / exp_times
    num = np.sum(w * img_stack * inv_exp_times, axis=0)
    den = np.sum(w, axis=0)
    img_hdr = np.divide(num, den, out=np.zeros_like(num, dtype=dtype), where=den > 0)

    # Fully under/overexposed pixel locations (all below or all above window)
    # fall back to conservative bounds.
    all_below = np.all(~valid_low, axis=0)
    all_above = np.all(~valid_high, axis=0)
    img_hdr[all_below] = z_min / np.max(exp_times)
    img_hdr[all_above] = z_max / np.min(exp_times)

    return img_hdr
