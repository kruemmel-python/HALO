"""High-level image processing extensions for HALO.

This module augments the low level HALO kernels with pure Python/Numpy
implementations of advanced operations that are commonly requested by users.
The implementation emphasises clarity and portability.  It supports a broad
set of dtypes (uint8/uint16/float32/float64) and gracefully degrades to
CPU-only execution when GPU hardware is not available.

The goal of the module is not to compete with specialised libraries such as
OpenCV, but to provide a convenient, dependency-light toolbox that can be used
for prototyping when the native HALO back-end does not yet expose a specific
kernel.  All functions operate on ``numpy.ndarray`` objects and return arrays of
matching dtype unless noted otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

try:  # NumPy is an optional dependency of the project.
    import numpy as np
except ImportError as exc:  # pragma: no cover - handled during runtime.
    raise RuntimeError(
        "halo_extensions requires NumPy. Install it via `pip install numpy`."
    ) from exc

ArrayLike = np.ndarray
ColorSpace = Literal["rgb", "gray", "hsv", "ycbcr"]
MorphOp = Literal[
    "erode",
    "dilate",
    "open",
    "close",
    "gradient",
    "tophat",
    "blackhat",
    "hitmiss",
]
Interpolation = Literal["nearest", "bilinear"]


def _ensure_float(image: ArrayLike, dtype: np.dtype) -> ArrayLike:
    """Return a floating representation without modifying the original array."""

    if image.dtype == dtype and image.flags.c_contiguous:
        return image
    return np.ascontiguousarray(image, dtype=dtype)


def _normalize_dtype(image: ArrayLike) -> Tuple[ArrayLike, float, float]:
    """Map integer images to [0, 1] while keeping track of the inverse mapping."""

    if np.issubdtype(image.dtype, np.floating):
        return image, 0.0, 1.0
    if image.dtype == np.uint8:
        scale = 255.0
    elif image.dtype == np.uint16:
        scale = 65535.0
    else:
        raise TypeError(f"Unsupported dtype: {image.dtype}")
    return image.astype(np.float32) / scale, 0.0, scale


# ---------------------------------------------------------------------------
# Advanced Image Processing Kernels
# ---------------------------------------------------------------------------

def bilateral_filter(
    image: ArrayLike,
    *,
    diameter: int = 5,
    sigma_color: float = 0.1,
    sigma_space: float = 2.0,
) -> ArrayLike:
    """Apply a bilateral filter to ``image``.

    Parameters
    ----------
    image:
        ``HxW`` or ``HxWxC`` array in uint8/uint16/float32/float64 format.
    diameter:
        Size of the square filter window (must be odd).
    sigma_color / sigma_space:
        Gaussian parameters for the range and spatial kernels.
    """

    if diameter % 2 == 0 or diameter <= 0:
        raise ValueError("diameter must be a positive odd integer")
    if sigma_color <= 0 or sigma_space <= 0:
        raise ValueError("sigma_color and sigma_space must be positive")

    src, mn, mx = _normalize_dtype(np.asarray(image))
    if src.ndim == 2:
        src = src[..., None]
    h, w, c = src.shape

    radius = diameter // 2
    grid_y, grid_x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    spatial_weights = np.exp(-(grid_x**2 + grid_y**2) / (2.0 * sigma_space**2))

    dst = np.zeros_like(src, dtype=np.float32)
    padded = np.pad(src, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")

    for y in range(h):
        for x in range(w):
            region = padded[y : y + diameter, x : x + diameter, :]
            center = padded[y + radius, x + radius, :]
            diff = region - center
            color_weights = np.exp(-(diff * diff) / (2.0 * sigma_color**2))
            weights = color_weights * spatial_weights[..., None]
            weights_sum = np.sum(weights, axis=(0, 1))
            weights_sum = np.maximum(weights_sum, 1e-8)
            dst[y, x] = np.sum(region * weights, axis=(0, 1)) / weights_sum

    dst = dst.reshape(h, w, c)
    if c == 1:
        dst = dst[..., 0]

    if mn == mx:
        return dst.astype(image.dtype)
    clipped = np.clip(dst * mx, 0.0, mx)
    return clipped.astype(image.dtype, copy=False)


def _gaussian_kernel(size: int, sigma: float) -> ArrayLike:
    radius = size // 2
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def _convolve_separable(image: ArrayLike, kernel: ArrayLike) -> ArrayLike:
    image_f = _ensure_float(image, np.float32)
    tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=image_f)
    dst = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=tmp)
    return dst


def canny_edge_detector(
    image: ArrayLike,
    *,
    low_threshold: float = 0.1,
    high_threshold: float = 0.3,
    gaussian_sigma: float = 1.4,
) -> ArrayLike:
    """Compute Canny edges for a grayscale or RGB image."""

    if high_threshold <= low_threshold:
        raise ValueError("high_threshold must be greater than low_threshold")
    if gaussian_sigma <= 0:
        raise ValueError("gaussian_sigma must be positive")

    if image.ndim == 3:
        gray = np.dot(image[..., :3], np.array([0.299, 0.587, 0.114], dtype=image.dtype))
    else:
        gray = image

    gray_norm, _, scale = _normalize_dtype(gray)
    kernel_size = int(gaussian_sigma * 6) | 1
    kernel = _gaussian_kernel(kernel_size, gaussian_sigma)
    smoothed = _convolve_separable(gray_norm, kernel)

    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    sobel_y = sobel_x.T
    grad_x = _convolve2d(smoothed, sobel_x)
    grad_y = _convolve2d(smoothed, sobel_y)
    magnitude = np.hypot(grad_x, grad_y)
    angle = np.arctan2(grad_y, grad_x)

    nms = _non_maximum_suppression(magnitude, angle)
    edges = _double_threshold_hysteresis(nms, low_threshold, high_threshold)
    return (edges * scale).astype(image.dtype, copy=False)


def _convolve2d(image: ArrayLike, kernel: ArrayLike) -> ArrayLike:
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = np.empty_like(image, dtype=np.float32)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded[y : y + kh, x : x + kw]
            out[y, x] = np.sum(region * kernel)
    return out


def _non_maximum_suppression(magnitude: ArrayLike, angle: ArrayLike) -> ArrayLike:
    angle = angle * (180.0 / np.pi)
    angle[angle < 0] += 180
    h, w = magnitude.shape
    output = np.zeros((h, w), dtype=np.float32)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            q = 0.0
            r = 0.0
            a = angle[y, x]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q = magnitude[y, x + 1]
                r = magnitude[y, x - 1]
            elif 22.5 <= a < 67.5:
                q = magnitude[y + 1, x - 1]
                r = magnitude[y - 1, x + 1]
            elif 67.5 <= a < 112.5:
                q = magnitude[y + 1, x]
                r = magnitude[y - 1, x]
            elif 112.5 <= a < 157.5:
                q = magnitude[y - 1, x - 1]
                r = magnitude[y + 1, x + 1]
            if magnitude[y, x] >= q and magnitude[y, x] >= r:
                output[y, x] = magnitude[y, x]
    return output


def _double_threshold_hysteresis(image: ArrayLike, low: float, high: float) -> ArrayLike:
    high_val = image.max() * high
    low_val = high_val * low
    res = np.zeros_like(image, dtype=np.float32)
    strong = image >= high_val
    weak = (image >= low_val) & ~strong
    res[strong] = 1.0

    from collections import deque

    q = deque(zip(*np.nonzero(strong)))
    while q:
        y, x = q.popleft()
        for ny in range(max(0, y - 1), min(image.shape[0], y + 2)):
            for nx in range(max(0, x - 1), min(image.shape[1], x + 2)):
                if weak[ny, nx] and res[ny, nx] == 0.0:
                    res[ny, nx] = 1.0
                    q.append((ny, nx))
    return res


# ---------------------------------------------------------------------------
# Colour Space Conversions
# ---------------------------------------------------------------------------

def convert_colorspace(image: ArrayLike, src: ColorSpace, dst: ColorSpace) -> ArrayLike:
    """Convert ``image`` between supported colour spaces."""

    src = src.lower()
    dst = dst.lower()
    if src == dst:
        return image.copy()

    if src == "rgb" and dst == "gray":
        return np.dot(image[..., :3], np.array([0.299, 0.587, 0.114], dtype=image.dtype))
    if src == "gray" and dst == "rgb":
        return np.repeat(image[..., None], 3, axis=2)

    if src == "rgb" and dst == "hsv":
        return _rgb_to_hsv(image)
    if src == "hsv" and dst == "rgb":
        return _hsv_to_rgb(image)
    if src == "rgb" and dst == "ycbcr":
        return _rgb_to_ycbcr(image)
    if src == "ycbcr" and dst == "rgb":
        return _ycbcr_to_rgb(image)

    raise ValueError(f"Unsupported conversion: {src}->{dst}")


def _rgb_to_hsv(image: ArrayLike) -> ArrayLike:
    image = _ensure_float(image, np.float32)
    maxc = image.max(axis=-1)
    minc = image.min(axis=-1)
    v = maxc
    delta = maxc - minc
    s = np.divide(delta, maxc, out=np.zeros_like(delta), where=maxc != 0)

    h = np.zeros_like(maxc)
    mask = delta != 0
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    idx = (maxc == r) & mask
    h[idx] = (g[idx] - b[idx]) / delta[idx]
    idx = (maxc == g) & mask
    h[idx] = 2.0 + (b[idx] - r[idx]) / delta[idx]
    idx = (maxc == b) & mask
    h[idx] = 4.0 + (r[idx] - g[idx]) / delta[idx]
    h = (h / 6.0) % 1.0
    return np.stack([h, s, v], axis=-1)


def _hsv_to_rgb(image: ArrayLike) -> ArrayLike:
    h, s, v = image[..., 0], image[..., 1], image[..., 2]
    i = np.floor(h * 6.0).astype(int)
    f = h * 6.0 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    i_mod = i % 6
    rgb = np.zeros(image.shape, dtype=image.dtype)
    idx = i_mod == 0
    rgb[idx] = np.stack([v[idx], t[idx], p[idx]], axis=-1)
    idx = i_mod == 1
    rgb[idx] = np.stack([q[idx], v[idx], p[idx]], axis=-1)
    idx = i_mod == 2
    rgb[idx] = np.stack([p[idx], v[idx], t[idx]], axis=-1)
    idx = i_mod == 3
    rgb[idx] = np.stack([p[idx], q[idx], v[idx]], axis=-1)
    idx = i_mod == 4
    rgb[idx] = np.stack([t[idx], p[idx], v[idx]], axis=-1)
    idx = i_mod == 5
    rgb[idx] = np.stack([v[idx], p[idx], q[idx]], axis=-1)
    return rgb


def _rgb_to_ycbcr(image: ArrayLike) -> ArrayLike:
    matrix = np.array(
        [
            [0.299, 0.587, 0.114],
            [-0.168736, -0.331264, 0.5],
            [0.5, -0.418688, -0.081312],
        ],
        dtype=np.float32,
    )
    shift = np.array([0.0, 0.5, 0.5], dtype=np.float32)
    result = image @ matrix.T + shift
    return result


def _ycbcr_to_rgb(image: ArrayLike) -> ArrayLike:
    matrix = np.array(
        [
            [1.0, 0.0, 1.402],
            [1.0, -0.344136, -0.714136],
            [1.0, 1.772, 0.0],
        ],
        dtype=np.float32,
    )
    shift = np.array([0.0, -0.5, -0.5], dtype=np.float32)
    result = (image + shift) @ matrix.T
    return np.clip(result, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Morphological Operations
# ---------------------------------------------------------------------------

def _structuring_element(size: int) -> ArrayLike:
    if size <= 0 or size % 2 == 0:
        raise ValueError("Structuring element size must be a positive odd integer")
    radius = size // 2
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x**2 + y**2 <= radius**2
    return mask.astype(np.uint8)


def morphological_operation(
    image: ArrayLike,
    operation: MorphOp,
    *,
    kernel: Optional[ArrayLike] = None,
    iterations: int = 1,
) -> ArrayLike:
    """Apply morphological operations channel-wise."""

    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if kernel is None:
        kernel = _structuring_element(3)
    kernel = kernel.astype(bool)

    src, mn, mx = _normalize_dtype(np.asarray(image))
    if src.ndim == 2:
        src = src[..., None]
    current = src.copy()

    for _ in range(iterations):
        if operation == "erode":
            current = _morph_erode(current, kernel)
        elif operation == "dilate":
            current = _morph_dilate(current, kernel)
        elif operation == "open":
            current = _morph_dilate(_morph_erode(current, kernel), kernel)
        elif operation == "close":
            current = _morph_erode(_morph_dilate(current, kernel), kernel)
        elif operation == "gradient":
            dil = _morph_dilate(current, kernel)
            ero = _morph_erode(current, kernel)
            current = dil - ero
        elif operation == "tophat":
            opened = _morph_dilate(_morph_erode(current, kernel), kernel)
            current = src - opened
        elif operation == "blackhat":
            closed = _morph_erode(_morph_dilate(current, kernel), kernel)
            current = closed - src
        elif operation == "hitmiss":
            current = _morph_hit_or_miss(current, kernel)
        else:  # pragma: no cover - exhaustive guard
            raise ValueError(f"Unsupported operation: {operation}")

    dst = current.squeeze()
    if mn == mx:
        return dst.astype(image.dtype, copy=False)
    return np.clip(dst * mx, 0.0, mx).astype(image.dtype, copy=False)


def _morph_erode(image: ArrayLike, kernel: ArrayLike) -> ArrayLike:
    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="edge")
    coords = np.argwhere(kernel)
    dst = np.empty_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded[y : y + kernel.shape[0], x : x + kernel.shape[1], :]
            masked = region[coords[:, 0], coords[:, 1], :]
            dst[y, x] = masked.min(axis=0)
    return dst


def _morph_dilate(image: ArrayLike, kernel: ArrayLike) -> ArrayLike:
    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="edge")
    coords = np.argwhere(kernel)
    dst = np.empty_like(image)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded[y : y + kernel.shape[0], x : x + kernel.shape[1], :]
            masked = region[coords[:, 0], coords[:, 1], :]
            dst[y, x] = masked.max(axis=0)
    return dst


def _morph_hit_or_miss(image: ArrayLike, kernel: ArrayLike) -> ArrayLike:
    binary = image > 0.5
    pad_y, pad_x = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(binary, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="constant")
    coords = np.argwhere(kernel)
    dst = np.zeros_like(binary, dtype=np.float32)
    for y in range(binary.shape[0]):
        for x in range(binary.shape[1]):
            region = padded[y : y + kernel.shape[0], x : x + kernel.shape[1], :]
            matches = np.all(region[coords[:, 0], coords[:, 1], :], axis=0)
            dst[y, x] = matches.astype(np.float32)
    return dst


# ---------------------------------------------------------------------------
# Geometry Operations
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AffineTransform:
    matrix: ArrayLike  # 3x3 homogeneous matrix

    @classmethod
    def from_components(
        cls,
        translation: Tuple[float, float] = (0.0, 0.0),
        rotation_deg: float = 0.0,
        scale: Tuple[float, float] = (1.0, 1.0),
        shear: Tuple[float, float] = (0.0, 0.0),
    ) -> "AffineTransform":
        tx, ty = translation
        sx, sy = scale
        shx, shy = shear
        theta = math.radians(rotation_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        matrix = np.array(
            [
                [sx * cos_t, -sy * sin_t + shx, tx],
                [sx * sin_t + shy, sy * cos_t, ty],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return cls(matrix)


def warp_affine(
    image: ArrayLike,
    matrix: ArrayLike,
    *,
    output_shape: Optional[Tuple[int, int]] = None,
    interpolation: Interpolation = "bilinear",
    cval: float = 0.0,
) -> ArrayLike:
    """Apply an affine warp to ``image`` using homogeneous coordinates."""

    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError("matrix must be 3x3 homogeneous")

    if output_shape is None:
        output_shape = image.shape[:2]
    out_h, out_w = output_shape

    inv = np.linalg.inv(matrix)
    grid_y, grid_x = np.mgrid[0:out_h, 0:out_w]
    ones = np.ones_like(grid_x)
    coords = np.stack([grid_x, grid_y, ones], axis=-1).reshape(-1, 3)
    mapped = coords @ inv.T
    mapped_x = mapped[:, 0] / mapped[:, 2]
    mapped_y = mapped[:, 1] / mapped[:, 2]

    warped = _sample(image, mapped_x, mapped_y, interpolation, cval)
    return warped.reshape(out_h, out_w, *image.shape[2:])


def warp_perspective(
    image: ArrayLike,
    matrix: ArrayLike,
    *,
    output_shape: Optional[Tuple[int, int]] = None,
    interpolation: Interpolation = "bilinear",
    cval: float = 0.0,
) -> ArrayLike:
    """Apply a perspective transform using a 3x3 homography."""

    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError("matrix must be 3x3")
    if output_shape is None:
        output_shape = image.shape[:2]
    out_h, out_w = output_shape
    grid_y, grid_x = np.mgrid[0:out_h, 0:out_w]
    ones = np.ones_like(grid_x)
    coords = np.stack([grid_x, grid_y, ones], axis=-1).reshape(-1, 3)
    mapped = coords @ np.linalg.inv(matrix).T
    mapped_x = mapped[:, 0] / mapped[:, 2]
    mapped_y = mapped[:, 1] / mapped[:, 2]
    warped = _sample(image, mapped_x, mapped_y, interpolation, cval)
    return warped.reshape(out_h, out_w, *image.shape[2:])


def warp_custom(
    image: ArrayLike,
    map_x: ArrayLike,
    map_y: ArrayLike,
    *,
    interpolation: Interpolation = "bilinear",
    cval: float = 0.0,
) -> ArrayLike:
    """Warp an image using explicit mapping arrays."""

    map_x = np.asarray(map_x, dtype=np.float32)
    map_y = np.asarray(map_y, dtype=np.float32)
    if map_x.shape != map_y.shape:
        raise ValueError("map_x and map_y must have identical shape")

    coords = np.stack([map_x.ravel(), map_y.ravel()], axis=-1)
    warped = _sample(image, coords[:, 0], coords[:, 1], interpolation, cval)
    return warped.reshape(map_x.shape + image.shape[2:])


def _sample(
    image: ArrayLike,
    xs: ArrayLike,
    ys: ArrayLike,
    interpolation: Interpolation,
    cval: float,
) -> ArrayLike:
    image = np.asarray(image)
    h, w = image.shape[:2]
    xs = xs.reshape(-1)
    ys = ys.reshape(-1)

    if interpolation == "nearest":
        xi = np.rint(xs).astype(int)
        yi = np.rint(ys).astype(int)
        mask = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        result = np.full((xs.size,) + image.shape[2:], cval, dtype=image.dtype)
        result[mask] = image[yi[mask], xi[mask]]
        return result

    if interpolation != "bilinear":
        raise ValueError("Unsupported interpolation method")

    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = xs - x0
    wy = ys - y0

    channel_shape = image.shape[2:]
    result = np.full((xs.size,) + channel_shape, cval, dtype=np.float32)

    def _expand(weights: ArrayLike) -> ArrayLike:
        if channel_shape:
            return weights.reshape((-1,) + (1,) * len(channel_shape))
        return weights

    def sample(ix: np.ndarray, iy: np.ndarray) -> ArrayLike:
        mask = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        out = np.zeros_like(result)
        if channel_shape:
            out[mask] = image[iy[mask], ix[mask]]
        else:
            out[mask] = image[iy[mask], ix[mask]]
        return out

    Ia = sample(x0, y0)
    Ib = sample(x1, y0)
    Ic = sample(x0, y1)
    Id = sample(x1, y1)

    w00 = _expand((1 - wx) * (1 - wy))
    w10 = _expand(wx * (1 - wy))
    w01 = _expand((1 - wx) * wy)
    w11 = _expand(wx * wy)

    result = Ia * w00 + Ib * w10 + Ic * w01 + Id * w11
    return result.astype(image.dtype, copy=False)


__all__ = [
    "bilateral_filter",
    "canny_edge_detector",
    "convert_colorspace",
    "morphological_operation",
    "warp_affine",
    "warp_perspective",
    "warp_custom",
    "AffineTransform",
]
