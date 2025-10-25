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
        backend = self.backend
        img = self._as_backend_array(image)
        ker = self._as_backend_array(kernel)
        ker = ker[::-1, ::-1]  # convolution flips the kernel
        pad_y, pad_x = ker.shape[0] // 2, ker.shape[1] // 2
        pad_width = ((pad_y, pad_y), (pad_x, pad_x)) + (() if img.ndim == 2 else ((0, 0),))
        padded = backend.pad(img, pad_width, mode="edge")
        out = backend.zeros_like(img)
        for y in range(out.shape[0]):
            for x in range(out.shape[1]):
                region = padded[y : y + ker.shape[0], x : x + ker.shape[1]]
                out[y, x] = backend.sum(region * ker)
        return self._to_numpy(out)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def affine_transform(self, image: ArrayLike, matrix: ArrayLike) -> ArrayLike:
        return warp_affine(image, matrix)


__all__ = ["HALOGPU"]
