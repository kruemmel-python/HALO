# halo.py (v0.4) — Python 3.12 Wrapper für HALO v0.3 + 2D-Image-Kern
# -------------------------------------------------------------------
# Neu:
#   - img_u8_to_f32_lut_axpby_2d(): 2D-Imagepfad mit Strides
#   - Hilfsfunktionen für LUT-Prüfung und Stride-Validierung
#
# Erwartete C-API in der DLL (zusätzlich zu v0.3):
#   int halo_img_u8_to_f32_lut_axpby(const unsigned char* src, long long src_stride,
#                                    float* dst, long long dst_stride,
#                                    int width, int height,
#                                    const float* lut256,
#                                    float scale, float offset,
#                                    float alpha, float beta,
#                                    int use_mt);

from __future__ import annotations
import ctypes as C
import json, os, platform
from pathlib import Path
from typing import Union, Sequence
from array import array

__all__ = ["HALO", "Impl", "make_identity_lut"]

# ---------------- Laden der DLL/.so ----------------
def _load_lib() -> C.CDLL:
    here = Path(__file__).resolve().parent
    if os.name == "nt":
        cand = here / "halo_fastpath.dll"
        return C.CDLL(str(cand)) if cand.exists() else C.CDLL("halo_fastpath.dll")
    else:
        cand = here / "libhalo_fastpath.so"
        return C.CDLL(str(cand)) if cand.exists() else C.CDLL("libhalo_fastpath.so")

_lib = _load_lib()

# ---------------- ctypes-Signaturen ----------------
_lib.halo_init_features.restype = C.c_int

_lib.halo_query_features.argtypes = [C.POINTER(C.c_int), C.POINTER(C.c_int), C.POINTER(C.c_int)]
_lib.halo_query_features.restype  = C.c_int

_lib.halo_configure.argtypes = [C.c_int, C.c_longlong, C.c_int]
_lib.halo_configure.restype  = C.c_int

_lib.halo_autotune.argtypes = [C.c_longlong, C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_int)]
_lib.halo_autotune.restype  = C.c_int

_lib.halo_set_impls.argtypes = [C.c_int, C.c_int]
_lib.halo_set_impls.restype  = C.c_int

_lib.halo_saxpy_f32.argtypes = [C.c_float, C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_longlong]
_lib.halo_saxpy_f32.restype  = C.c_int

_lib.halo_saxpy_f32_mt.argtypes = [C.c_float, C.POINTER(C.c_float), C.POINTER(C.c_float), C.c_longlong]
_lib.halo_saxpy_f32_mt.restype  = C.c_int

_lib.halo_sum_f32.argtypes = [C.POINTER(C.c_float), C.c_longlong, C.POINTER(C.c_float)]
_lib.halo_sum_f32.restype  = C.c_int

# NEU: 2D-Image-Kern
_lib.halo_img_u8_to_f32_lut_axpby.argtypes = [
    C.POINTER(C.c_ubyte), C.c_longlong,           # src, src_stride (bytes)
    C.POINTER(C.c_float), C.c_longlong,           # dst, dst_stride (bytes)
    C.c_int, C.c_int,                              # width, height
    C.POINTER(C.c_float),                          # lut256[256]
    C.c_float, C.c_float,                          # scale, offset
    C.c_float, C.c_float,                          # alpha, beta
    C.c_int                                        # use_mt (0/1)
]
_lib.halo_img_u8_to_f32_lut_axpby.restype = C.c_int

# ---------------- Profil-Persistenz ----------------
PROFILE_PATH = Path.home() / ".halo_profile.json"

def _read_profile() -> dict:
    if PROFILE_PATH.exists():
        try:
            return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _write_profile(d: dict) -> None:
    PROFILE_PATH.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")

# ---------------- Typen & Helfer ----------------
ArrayLikeFloat = Union[array, memoryview]
ArrayLikeU8    = Union[bytes, bytearray, memoryview]

def _to_c_float_ptr(buf: ArrayLikeFloat) -> tuple[C.POINTER(C.c_float), int]:
    """array('f') oder schreibbarer memoryview('f') → Zero-Copy ctypes-Pointer + Länge (1D)."""
    match buf:
        case arr if isinstance(arr, array) and arr.typecode == 'f':
            n = len(arr)
            c_arr = (C.c_float * n).from_buffer(arr)
            return C.cast(c_arr, C.POINTER(C.c_float)), n
        case mv if isinstance(mv, memoryview) and mv.format == 'f' and mv.contiguous and not mv.readonly:
            n = mv.shape[0] if hasattr(mv, "shape") and mv.ndim == 1 else (mv.nbytes // C.sizeof(C.c_float))
            c_arr = (C.c_float * n).from_buffer(mv)
            return C.cast(c_arr, C.POINTER(C.c_float)), n
        case _:
            raise TypeError("Erwarte array('f') oder schreibbaren, kontiguösen memoryview('f').")

def _to_c_u8_ptr_2d(buf: ArrayLikeU8, width: int, height: int, stride_bytes: int) -> C.POINTER(C.c_ubyte):
    """bytes/bytearray/memoryview('B'/'b') als 2D-Quellpuffer mit gegebenem Stride (in Bytes)."""
    if isinstance(buf, (bytes, bytearray)):
        mv = memoryview(buf)
    elif isinstance(buf, memoryview):
        mv = buf
    else:
        raise TypeError("src erwartet bytes, bytearray oder memoryview.")
    # Format check: 'B' (unsigned) oder 'b' (signed) sind 1-Byte; wir casten auf unsigned.
    if mv.itemsize != 1 or not mv.contiguous:
        raise TypeError("src muss kontiguös und 1-Byte-Elemente haben.")
    # Kapazitätscheck (konservativ)
    min_bytes = (height - 1) * stride_bytes + width
    if mv.nbytes < min_bytes:
        raise ValueError(f"src: zu klein für height={height}, stride={stride_bytes}, width={width}.")
    c_arr = (C.c_ubyte * mv.nbytes).from_buffer(mv) if not mv.readonly else (C.c_ubyte * mv.nbytes).from_buffer_copy(mv)
    return C.cast(c_arr, C.POINTER(C.c_ubyte))

def _to_c_f32_ptr_2d(buf: ArrayLikeFloat, width: int, height: int, stride_bytes: int) -> C.POINTER(C.c_float):
    """array('f') / memoryview('f') als 2D-Zielpuffer mit vorgegebenem Stride (Bytes)."""
    match buf:
        case arr if isinstance(arr, array) and arr.typecode == 'f':
            mv = memoryview(arr)
        case mv if isinstance(mv, memoryview) and mv.format == 'f' and mv.contiguous and not mv.readonly:
            pass
        case _:
            raise TypeError("dst erwartet array('f') oder schreibbaren, kontiguösen memoryview('f').")
    # Kapazitätscheck: stride_bytes ist Bytes/Zeile
    min_bytes = (height - 1) * stride_bytes + width * 4
    if mv.nbytes < min_bytes:
        raise ValueError(f"dst: zu klein für height={height}, stride={stride_bytes}, width={width}.")
    c_arr = (C.c_float * (mv.nbytes // 4)).from_buffer(mv)  # Zero-Copy
    return C.cast(c_arr, C.POINTER(C.c_float))

def _to_c_lut_ptr(lut: Union[Sequence[float], array, memoryview]) -> C.POINTER(C.c_float):
    """LUT muss 256 float-Werte enthalten."""
    if isinstance(lut, array) and lut.typecode == 'f' and len(lut) == 256:
        c_arr = (C.c_float * 256).from_buffer(lut)
        return C.cast(c_arr, C.POINTER(C.c_float))
    if isinstance(lut, memoryview) and lut.format == 'f' and lut.contiguous and len(lut) == 256:
        c_arr = (C.c_float * 256).from_buffer(lut)  # Zero-Copy
        return C.cast(c_arr, C.POINTER(C.c_float))
    if isinstance(lut, Sequence) and len(lut) == 256:
        arrf = array('f', lut)
        c_arr = (C.c_float * 256).from_buffer(arrf)
        return C.cast(c_arr, C.POINTER(C.c_float))
    raise ValueError("LUT muss genau 256 float-Werte enthalten (array('f'), memoryview('f') oder Sequenz).")

def make_identity_lut() -> array:
    """Erzeugt LUT[i] = float(i)."""
    return array('f', [float(i) for i in range(256)])

class Impl:
    Scalar      = 0
    SSE2        = 1
    AVX2        = 2
    AVX2_STREAM = 3

# ---------------- Haupt-Klasse ----------------
class HALO:
    """
    HALO v0.4 – inklusive 2D-Image-Kern mit Strides.
    """
    def __init__(
        self,
        n_autotune: int = 1_000_000,
        iters: int = 10,
        force_autotune: bool = False,
        enable_streaming: bool = True,
        stream_threshold: int = 1_000_000,
        threads: int = 1,
    ):
        if _lib.halo_init_features() != 0:
            raise RuntimeError("halo_init_features() fehlgeschlagen.")
        self.features = self._query_features()

        self.configure(enable_streaming=enable_streaming, stream_threshold=stream_threshold, threads=threads)

        self.profile = _read_profile()
        need_tune = force_autotune or ("saxpy_impl" not in self.profile) or ("sum_impl" not in self.profile)

        if need_tune:
            sax = C.c_int(0); s = C.c_int(0)
            if _lib.halo_autotune(C.c_longlong(n_autotune), C.c_int(iters), C.byref(sax), C.byref(s)) != 0:
                raise RuntimeError("halo_autotune() fehlgeschlagen.")
            self.profile = {
                "cpu": {
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                    "platform": platform.platform(),
                },
                "features": self.features,
                "saxpy_impl": sax.value,
                "sum_impl":   s.value,
                "cfg": {"streaming": bool(enable_streaming), "thr": int(stream_threshold), "threads": int(threads)},
            }
            _write_profile(self.profile)
        else:
            _lib.halo_set_impls(C.c_int(self.profile["saxpy_impl"]), C.c_int(self.profile["sum_impl"]))

    # ---- Public API (bestehend) ----
    def configure(self, enable_streaming: bool, stream_threshold: int, threads: int) -> None:
        if _lib.halo_configure(C.c_int(1 if enable_streaming else 0),
                               C.c_longlong(stream_threshold),
                               C.c_int(threads)) != 0:
            raise RuntimeError("halo_configure() fehlgeschlagen.")

    def saxpy(self, a: float, x: ArrayLikeFloat, y: ArrayLikeFloat) -> None:
        xptr, n1 = _to_c_float_ptr(x)
        yptr, n2 = _to_c_float_ptr(y)
        if n1 != n2:
            raise ValueError("x und y müssen gleiche Länge haben.")
        if _lib.halo_saxpy_f32(C.c_float(a), xptr, yptr, C.c_longlong(n1)) != 0:
            raise RuntimeError("halo_saxpy_f32() fehlgeschlagen.")

    def saxpy_mt(self, a: float, x: ArrayLikeFloat, y: ArrayLikeFloat) -> None:
        xptr, n1 = _to_c_float_ptr(x)
        yptr, n2 = _to_c_float_ptr(y)
        if n1 != n2:
            raise ValueError("x und y müssen gleiche Länge haben.")
        if _lib.halo_saxpy_f32_mt(C.c_float(a), xptr, yptr, C.c_longlong(n1)) != 0:
            raise RuntimeError("halo_saxpy_f32_mt() fehlgeschlagen.")

    def sum(self, x: ArrayLikeFloat) -> float:
        xptr, n = _to_c_float_ptr(x)
        out = C.c_float(0.0)
        if _lib.halo_sum_f32(xptr, C.c_longlong(n), C.byref(out)) != 0:
            raise RuntimeError("halo_sum_f32() fehlgeschlagen.")
        return float(out.value)

    def set_impls(self, saxpy_impl: int, sum_impl: int, persist: bool = False) -> None:
        if _lib.halo_set_impls(C.c_int(saxpy_impl), C.c_int(sum_impl)) != 0:
            raise RuntimeError("halo_set_impls() fehlgeschlagen.")
        if persist:
            self.profile["saxpy_impl"] = saxpy_impl
            self.profile["sum_impl"]   = sum_impl
            _write_profile(self.profile)

    # ---- Neu: 2D-Image-Kern ----
    def img_u8_to_f32_lut_axpby_2d(
        self,
        src: ArrayLikeU8,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst: ArrayLikeFloat,
        dst_stride_bytes: int,
        lut256: Union[Sequence[float], array, memoryview],
        *,
        scale: float = 1.0,
        offset: float = 0.0,
        alpha: float = 0.0,
        beta: float = 1.0,
        use_mt: bool = True,
    ) -> None:
        """
        Wendet auf ein 2D-uint8-Bild eine LUT + Affintransformation an und mischt in dst:

            tmp = lut[src] * scale + offset
            dst = alpha * dst + beta * tmp

        Parameter:
            src: bytes/bytearray/memoryview (1 Byte pro Pixel), Größe mindestens height*stride
            width, height: Bilddimensionen (Pixel)
            src_stride_bytes: Byte-Stride pro src-Zeile (>= width)
            dst: array('f') oder memoryview('f'), float32-Zielpuffer
            dst_stride_bytes: Bytes pro dst-Zeile (>= width*4)
            lut256: 256 floats (array('f') empfohlen)
            scale, offset, alpha, beta: Skalare
            use_mt: True → Multi-Thread, False → Single-Thread
        """
        if width <= 0 or height <= 0:
            raise ValueError("width/height müssen > 0 sein.")
        if src_stride_bytes < width:
            raise ValueError("src_stride_bytes < width (Bytes) ist ungültig.")
        if dst_stride_bytes < width * 4:
            raise ValueError("dst_stride_bytes < width*4 (Bytes) ist ungültig.")

        c_src = _to_c_u8_ptr_2d(src, width, height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, width, height, dst_stride_bytes)
        c_lut = _to_c_lut_ptr(lut256)

        rc = _lib.halo_img_u8_to_f32_lut_axpby(
            c_src, C.c_longlong(src_stride_bytes),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(width), C.c_int(height),
            c_lut,
            C.c_float(scale), C.c_float(offset),
            C.c_float(alpha), C.c_float(beta),
            C.c_int(1 if use_mt else 0)
        )
        if rc != 0:
            raise RuntimeError(f"halo_img_u8_to_f32_lut_axpby() fehlgeschlagen (rc={rc}).")

    # ---- intern ----
    def _query_features(self) -> dict:
        sse2 = C.c_int(0); avx2 = C.c_int(0); avx512 = C.c_int(0)
        if _lib.halo_query_features(C.byref(sse2), C.byref(avx2), C.byref(avx512)) != 0:
            raise RuntimeError("halo_query_features() fehlgeschlagen.")
        return {"sse2": bool(sse2.value), "avx2": bool(avx2.value), "avx512": bool(avx512.value)}
