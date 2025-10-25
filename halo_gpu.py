"""GPU acceleration helpers for HALO.

The native HALO runtime focuses on highly optimised CPU kernels.  This module
provides an optional GPU back-end implemented in pure Python.  It auto-detects
available frameworks (currently CUDA via CuPy) and falls back to NumPy when the
GPU toolchain is missing.  The API mirrors a small subset of the HALO CPU class
and can be extended gradually as more kernels are required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:  # pragma: no cover - GPU dependencies might not be installed.
    import cupy as cp
    _HAS_CUPY = True
except Exception:  # pragma: no cover - fallback path exercised in CPU environments.
    cp = None  # type: ignore
    _HAS_CUPY = False

import numpy as np

try:  # Support both package and flat-module imports.
    from .halo_extensions import warp_affine
except ImportError:  # pragma: no cover - flat import fallback.
    from halo_extensions import warp_affine  # type: ignore

ArrayLike = np.ndarray


@dataclass(slots=True)
class HALOGPU:
    """Light-weight GPU execution context.

    Parameters
    ----------
    prefer_gpu:
        If ``True`` (default) the class attempts to use CuPy.  When the backend
        is not available, it transparently falls back to NumPy.
    stream:
        Optional CuPy stream for asynchronous execution.
    """

    prefer_gpu: bool = True
    stream: Optional["cp.cuda.Stream"] = None

    def __post_init__(self) -> None:
        if self.prefer_gpu and not _HAS_CUPY:
            self.prefer_gpu = False

    # ------------------------------------------------------------------
    # Back-end helpers
    # ------------------------------------------------------------------
    @property
    def backend(self):  # type: ignore[override]
        if self.prefer_gpu and _HAS_CUPY:
            return cp
        return np

    def _as_backend_array(self, data: ArrayLike):
        if self.backend is np:
            return np.asarray(data)
        return cp.asarray(data)

    def _to_numpy(self, data):
        if self.backend is np:
            return np.asarray(data)
        return cp.asnumpy(data)

    def _float_dtype(self):
        backend = self.backend
        return backend.float32 if backend is not np else np.float32

    def _ensure_float_image(self, data: ArrayLike):
        arr = self._as_backend_array(data)
        dtype = self._float_dtype()
        if arr.dtype != dtype:
            arr = arr.astype(dtype)
        return arr

    def _apply_channelwise(self, image, fn):
        if image.ndim == 2:
            return fn(image)
        backend = self.backend
        channels = [fn(image[..., c]) for c in range(image.shape[2])]
        return backend.stack(channels, axis=2)

    def _convolve_single_channel(self, channel, kernel):
        backend = self.backend
        ker = self._as_backend_array(kernel)
        ker = ker[::-1, ::-1]
        pad_y, pad_x = ker.shape[0] // 2, ker.shape[1] // 2
        padded = backend.pad(channel, ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
        sliding = getattr(backend.lib.stride_tricks, "sliding_window_view", None)
        if sliding is not None:
            windows = sliding(padded, ker.shape)
            return (windows * ker).sum(axis=(-1, -2))
        out = backend.zeros_like(channel)
        for y in range(out.shape[0]):
            for x in range(out.shape[1]):
                region = padded[y : y + ker.shape[0], x : x + ker.shape[1]]
                out[y, x] = backend.sum(region * ker)
        return out

    # ------------------------------------------------------------------
    # Vector operations
    # ------------------------------------------------------------------
    def saxpy(self, a: float, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        bx = self._as_backend_array(x)
        by = self._as_backend_array(y)
        result = a * bx + by
        return self._to_numpy(result)

    def sum(self, x: ArrayLike) -> float:
        bx = self._as_backend_array(x)
        return float(bx.sum())

    # ------------------------------------------------------------------
    # Image helpers
    # ------------------------------------------------------------------
    def box_blur(self, image: ArrayLike, radius: int = 1) -> ArrayLike:
        if radius < 0:
            raise ValueError("radius must be non-negative")
        backend = self.backend
        kernel_size = radius * 2 + 1
        kernel = backend.ones((kernel_size, kernel_size), dtype=image.dtype)
        kernel /= kernel.size
        return self.convolve2d(image, kernel)

    def convolve2d(self, image: ArrayLike, kernel) -> ArrayLike:
        img = self._as_backend_array(image)
        ker = self._as_backend_array(kernel)
        result = self._apply_channelwise(img, lambda ch: self._convolve_single_channel(ch, ker))
        return self._to_numpy(result)

    def gaussian_kernel(self, sigma: float, radius: Optional[int] = None):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        backend = self.backend
        dtype = self._float_dtype()
        if radius is None:
            radius = max(1, int(round(3 * sigma)))
        ax = backend.arange(-radius, radius + 1, dtype=dtype)
        kernel = backend.exp(-(ax ** 2) / (2 * sigma * sigma))
        kernel = kernel[:, None] @ kernel[None, :]
        kernel /= kernel.sum()
        return kernel

    def gaussian_blur_f32(self, image: ArrayLike, sigma: float) -> ArrayLike:
        kernel = self.gaussian_kernel(sigma)
        img = self._ensure_float_image(image)
        result = self._apply_channelwise(img, lambda ch: self._convolve_single_channel(ch, kernel))
        return self._to_numpy(result)

    def sobel_f32(self, image: ArrayLike) -> ArrayLike:
        backend = self.backend
        dtype = self._float_dtype()
        gx = backend.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=dtype)
        gy = backend.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=dtype)
        img = self._ensure_float_image(image)

        def sobel_channel(ch):
            grad_x = self._convolve_single_channel(ch, gx)
            grad_y = self._convolve_single_channel(ch, gy)
            return backend.sqrt(grad_x * grad_x + grad_y * grad_y)

        result = self._apply_channelwise(img, sobel_channel)
        result = backend.clip(result, 0.0, 1.0)
        return self._to_numpy(result)

    def median3x3_f32(self, image: ArrayLike) -> ArrayLike:
        backend = self.backend
        img = self._ensure_float_image(image)

        def median_channel(ch):
            pad = backend.pad(ch, ((1, 1), (1, 1)), mode="edge")
            sliding = getattr(backend.lib.stride_tricks, "sliding_window_view", None)
            if sliding is not None:
                windows = sliding(pad, (3, 3))
                return backend.median(windows, axis=(-1, -2))
            out = backend.zeros_like(ch)
            for y in range(out.shape[0]):
                for x in range(out.shape[1]):
                    region = pad[y : y + 3, x : x + 3]
                    out[y, x] = backend.median(region)
            return out

        result = self._apply_channelwise(img, median_channel)
        return self._to_numpy(result)

    def unsharp_mask_f32(self, image: ArrayLike, sigma: float, amount: float, threshold: float) -> ArrayLike:
        backend = self.backend
        img = self._ensure_float_image(image)
        blurred = self._as_backend_array(self.gaussian_blur_f32(img, sigma))
        mask = img - blurred
        enhanced = img + amount * mask
        if threshold > 0:
            low_contrast = backend.abs(mask) < threshold
            enhanced = backend.where(low_contrast, img, enhanced)
        enhanced = backend.clip(enhanced, 0.0, 1.0)
        return self._to_numpy(enhanced)

    def invert_f32(self, image: ArrayLike, min_val: float = 0.0, max_val: float = 1.0) -> ArrayLike:
        backend = self.backend
        img = self._ensure_float_image(image)
        result = max_val + min_val - img
        return self._to_numpy(backend.clip(result, min_val, max_val))

    def gamma_f32(self, image: ArrayLike, gamma: float, gain: float) -> ArrayLike:
        backend = self.backend
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        img = self._ensure_float_image(image)
        adjusted = gain * backend.power(backend.clip(img, 0.0, 1.0), gamma)
        return self._to_numpy(backend.clip(adjusted, 0.0, 1.0))

    def levels_f32(
        self,
        image: ArrayLike,
        in_low: float,
        in_high: float,
        out_low: float,
        out_high: float,
        gamma: float,
    ) -> ArrayLike:
        backend = self.backend
        img = self._ensure_float_image(image)
        in_range = max(in_high - in_low, 1e-6)
        norm = backend.clip((img - in_low) / in_range, 0.0, 1.0)
        if gamma > 0 and gamma != 1.0:
            norm = backend.power(norm, gamma)
        result = out_low + norm * (out_high - out_low)
        return self._to_numpy(backend.clip(result, 0.0, 1.0))

    def threshold_f32(self, image: ArrayLike, low: float, high: float, low_value: float, high_value: float) -> ArrayLike:
        backend = self.backend
        img = self._ensure_float_image(image)
        result = backend.where(img < low, low_value, backend.where(img > high, high_value, img))
        return self._to_numpy(result)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def affine_transform(self, image: ArrayLike, matrix: ArrayLike) -> ArrayLike:
        return warp_affine(image, matrix)


__all__ = ["HALOGPU"]
