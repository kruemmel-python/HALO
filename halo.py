# halo.py (v0.5b) — Python 3.12 Wrapper für HALO v0.5b + 2D-Image-Kern
# - NEU: atexit-Hook ruft halo_shutdown_pool(), falls in DLL vorhanden.
# - Optional: HALO.close() für manuellen Shutdown.

from __future__ import annotations
import ctypes as C
import ctypes.util
import json, os, platform, atexit
from pathlib import Path
from typing import Union, Sequence, Tuple
from array import array

__all__ = [
    "HALO",
    "Impl",
    "make_identity_lut",
    "allocate_aligned_bytes",
    "allocate_aligned_f32",
    "pin_memory",
]

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

_lib.halo_img_u8_to_f32_lut_axpby.argtypes = [
    C.POINTER(C.c_ubyte), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.POINTER(C.c_float),
    C.c_float, C.c_float,
    C.c_float, C.c_float,
    C.c_int
]
_lib.halo_img_u8_to_f32_lut_axpby.restype = C.c_int

_lib.halo_img_u16_to_f32_axpby.argtypes = [
    C.POINTER(C.c_ushort), C.c_longlong,
    C.POINTER(C.c_float),  C.c_longlong,
    C.c_int, C.c_int,
    C.c_int,
    C.c_float, C.c_float,
    C.c_float, C.c_float,
    C.c_int
]
_lib.halo_img_u16_to_f32_axpby.restype = C.c_int

_lib.halo_img_rgb_u8_to_f32_interleaved.argtypes = [
    C.POINTER(C.c_ubyte), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_float, C.c_float,
    C.c_float, C.c_float,
    C.c_int
]
_lib.halo_img_rgb_u8_to_f32_interleaved.restype = C.c_int

_lib.halo_img_rgb_u8_to_f32_planar.argtypes = [
    C.POINTER(C.c_ubyte), C.c_longlong,
    C.POINTER(C.c_ubyte), C.c_longlong,
    C.POINTER(C.c_ubyte), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_float, C.c_float,
    C.c_float, C.c_float,
    C.c_int
]
_lib.halo_img_rgb_u8_to_f32_planar.restype = C.c_int

_lib.halo_img_box_blur_f32.argtypes = [
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_int,
    C.c_int
]
_lib.halo_img_box_blur_f32.restype = C.c_int

_lib.halo_img_gauss_blur_f32.argtypes = [
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_int,
    C.c_float,
    C.c_int
]
_lib.halo_img_gauss_blur_f32.restype = C.c_int

_lib.halo_img_sobel_f32.argtypes = [
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_int
]
_lib.halo_img_sobel_f32.restype = C.c_int

_lib.halo_img_resize_f32.argtypes = [
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_int,
    C.c_int
]
_lib.halo_img_resize_f32.restype = C.c_int

_lib.halo_img_relu_clamp_axpby_f32.argtypes = [
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_float, C.c_float,
    C.c_float, C.c_float,
    C.c_int,
    C.c_int
]
_lib.halo_img_relu_clamp_axpby_f32.restype = C.c_int

# Optional: Pool-Shutdown wenn vorhanden (v0.5b+)
try:
    _lib.halo_shutdown_pool.restype = None
    _HAS_SHUTDOWN = True
except AttributeError:
    _HAS_SHUTDOWN = False

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
ArrayLikeU16   = Union[array, bytearray, memoryview]

def _to_c_float_ptr(buf: ArrayLikeFloat) -> Tuple[C.POINTER(C.c_float), int]:
    match buf:
        case arr if isinstance(arr, array) and arr.typecode == 'f':
            n = len(arr)
            c_arr = (C.c_float * n).from_buffer(arr)
            return C.cast(c_arr, C.POINTER(C.c_float)), n
        case mv if isinstance(mv, memoryview) and mv.format == 'f' and mv.contiguous and not mv.readonly:
            n = mv.shape[0] if hasattr(mv, "shape") and getattr(mv, "ndim", 1) == 1 else (mv.nbytes // C.sizeof(C.c_float))
            c_arr = (C.c_float * (mv.nbytes // 4)).from_buffer(mv)
            return C.cast(c_arr, C.POINTER(C.c_float)), n
        case _:
            raise TypeError("Erwarte array('f') oder schreibbaren, kontiguösen memoryview('f').")

def _to_c_u8_ptr_2d(buf: ArrayLikeU8, width: int, height: int, stride_bytes: int) -> C.POINTER(C.c_ubyte):
    if isinstance(buf, (bytes, bytearray)):
        mv = memoryview(buf)
    elif isinstance(buf, memoryview):
        mv = buf
    else:
        raise TypeError("src erwartet bytes, bytearray oder memoryview.")
    if mv.itemsize != 1 or not mv.contiguous:
        raise TypeError("src muss kontiguös und 1-Byte-Elemente haben.")
    min_bytes = (height - 1) * stride_bytes + width
    if mv.nbytes < min_bytes:
        raise ValueError(f"src: zu klein für height={height}, stride={stride_bytes}, width={width}.")
    if mv.readonly:
        c_arr = (C.c_ubyte * mv.nbytes).from_buffer_copy(mv)
    else:
        c_arr = (C.c_ubyte * mv.nbytes).from_buffer(mv)
    return C.cast(c_arr, C.POINTER(C.c_ubyte))

def _to_c_f32_ptr_2d(buf: ArrayLikeFloat, width: int, height: int, stride_bytes: int) -> C.POINTER(C.c_float):
    match buf:
        case arr if isinstance(arr, array) and arr.typecode == 'f':
            mv = memoryview(arr)
        case mv if isinstance(mv, memoryview) and mv.format == 'f' and mv.contiguous and not mv.readonly:
            pass
        case _:
            raise TypeError("dst erwartet array('f') oder schreibbaren, kontiguösen memoryview('f').")
    min_bytes = (height - 1) * stride_bytes + width * 4
    if mv.nbytes < min_bytes:
        raise ValueError(f"dst: zu klein für height={height}, stride={stride_bytes}, width={width}.")
    c_arr = (C.c_float * (mv.nbytes // 4)).from_buffer(mv)
    return C.cast(c_arr, C.POINTER(C.c_float))

def _to_c_lut_ptr(lut: Union[Sequence[float], array, memoryview]) -> C.POINTER(C.c_float):
    if isinstance(lut, array) and lut.typecode == 'f' and len(lut) == 256:
        c_arr = (C.c_float * 256).from_buffer(lut)
        return C.cast(c_arr, C.POINTER(C.c_float))
    if isinstance(lut, memoryview) and lut.format == 'f' and lut.contiguous and len(lut) == 256:
        c_arr = (C.c_float * 256).from_buffer(lut)
        return C.cast(c_arr, C.POINTER(C.c_float))
    if isinstance(lut, Sequence) and len(lut) == 256:
        arrf = array('f', lut)
        c_arr = (C.c_float * 256).from_buffer(arrf)
        return C.cast(c_arr, C.POINTER(C.c_float))
    raise ValueError("LUT muss genau 256 float-Werte enthalten (array('f'), memoryview('f') oder Sequenz).")

def make_identity_lut() -> array:
    return array('f', [float(i) for i in range(256)])

def _to_c_u16_ptr_2d(buf: ArrayLikeU16, width: int, height: int, stride_bytes: int) -> C.POINTER(C.c_ushort):
    if isinstance(buf, array) and buf.typecode == 'H':
        mv = memoryview(buf)
    elif isinstance(buf, bytearray):
        mv = memoryview(buf)
    elif isinstance(buf, memoryview):
        mv = buf
    else:
        raise TypeError("src erwartet array('H'), bytearray oder memoryview('H').")
    if mv.readonly:
        raise ValueError("src darf nicht read-only sein.")
    if mv.itemsize not in (1, 2):
        raise TypeError("src muss 1- oder 2-Byte-Elemente haben.")
    if not mv.contiguous:
        raise TypeError("src muss kontiguös sein.")
    min_bytes = (height - 1) * stride_bytes + width * 2
    if mv.nbytes < min_bytes:
        raise ValueError(f"src: zu klein für height={height}, stride={stride_bytes}, width={width}.")
    mv16 = mv.cast('H') if mv.itemsize == 1 else mv.cast('H')
    buf_ref = (C.c_ushort * (mv16.nbytes // 2)).from_buffer(mv16)
    return C.cast(buf_ref, C.POINTER(C.c_ushort))

class AlignedBuffer:
    __slots__ = ("_raw", "_mv")

    def __init__(self, size: int, alignment: int = 64):
        if size <= 0:
            raise ValueError("size muss > 0 sein.")
        if alignment <= 0 or (alignment & (alignment - 1)) != 0:
            raise ValueError("alignment muss Potenz von 2 sein.")
        self._raw = bytearray(size + alignment)
        base = C.addressof(C.c_char.from_buffer(self._raw))
        offset = (alignment - (base % alignment)) % alignment
        self._mv = memoryview(self._raw)[offset:offset + size]

    @property
    def memoryview(self) -> memoryview:
        return self._mv

    @property
    def buffer(self) -> bytearray:
        return self._raw

def allocate_aligned_bytes(size: int, alignment: int = 64) -> memoryview:
    return AlignedBuffer(size, alignment).memoryview

def allocate_aligned_f32(length: int, alignment: int = 64) -> memoryview:
    mv = AlignedBuffer(length * 4, alignment).memoryview
    return mv.cast('f')

def pin_memory(buf: Union[memoryview, bytearray, array]) -> bool:
    mv = memoryview(buf)
    if mv.readonly:
        raise ValueError("Buffer darf nicht read-only sein.")
    if not mv.contiguous:
        raise ValueError("Buffer muss kontiguös sein.")
    mv_bytes = mv.cast('B')
    address = C.addressof(C.c_char.from_buffer(mv_bytes))
    size = mv.nbytes
    try:
        if os.name == "nt":
            kernel32 = C.windll.kernel32
            res = kernel32.VirtualLock(C.c_void_p(address), C.c_size_t(size))
            return bool(res)
        else:
            libc_name = ctypes.util.find_library("c") or "libc.so.6"
            libc = C.CDLL(libc_name)
            res = libc.mlock(C.c_void_p(address), C.c_size_t(size))
            return res == 0
    except Exception:
        return False

class Impl:
    Scalar      = 0
    SSE2        = 1
    AVX2        = 2
    AVX2_STREAM = 3

# ---------------- Haupt-Klasse ----------------
class HALO:
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

        self.configure(enable_streaming=enable_streaming,
                       stream_threshold=stream_threshold,
                       threads=threads)

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
            if _lib.halo_set_impls(C.c_int(self.profile["saxpy_impl"]),
                                   C.c_int(self.profile["sum_impl"])) != 0:
                raise RuntimeError("halo_set_impls() fehlgeschlagen.")

    def configure(self, enable_streaming: bool, stream_threshold: int, threads: int) -> None:
        if _lib.halo_configure(C.c_int(1 if enable_streaming else 0),
                               C.c_longlong(stream_threshold),
                               C.c_int(threads)) != 0:
            raise RuntimeError("halo_configure() fehlgeschlagen.")

    def set_threads(self, threads: int) -> None:
        cfg = self.profile.get("cfg", {})
        streaming = bool(cfg.get("streaming", True))
        thr       = int(cfg.get("thr", 1_000_000))
        self.configure(enable_streaming=streaming, stream_threshold=thr, threads=threads)
        cfg.update({"threads": int(threads)})
        self.profile["cfg"] = cfg
        _write_profile(self.profile)

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

    def img_u16_to_f32_axpby_2d(
        self,
        src: ArrayLikeU16,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst: ArrayLikeFloat,
        dst_stride_bytes: int,
        *,
        bits_per_sample: int = 10,
        scale: float = 1.0,
        offset: float = 0.0,
        alpha: float = 0.0,
        beta: float = 1.0,
        use_mt: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width/height müssen > 0 sein.")
        if src_stride_bytes < width * 2:
            raise ValueError("src_stride_bytes < width*2 (Bytes) ist ungültig.")
        if dst_stride_bytes < width * 4:
            raise ValueError("dst_stride_bytes < width*4 (Bytes) ist ungültig.")
        c_src = _to_c_u16_ptr_2d(src, width, height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, width, height, dst_stride_bytes)
        rc = _lib.halo_img_u16_to_f32_axpby(
            c_src, C.c_longlong(src_stride_bytes),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(width), C.c_int(height),
            C.c_int(bits_per_sample),
            C.c_float(scale), C.c_float(offset),
            C.c_float(alpha), C.c_float(beta),
            C.c_int(1 if use_mt else 0)
        )
        if rc != 0:
            raise RuntimeError(f"halo_img_u16_to_f32_axpby() fehlgeschlagen (rc={rc}).")

    def img_rgb_u8_to_f32_interleaved_2d(
        self,
        src: ArrayLikeU8,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst_r: ArrayLikeFloat,
        dst_stride_r: int,
        dst_g: ArrayLikeFloat,
        dst_stride_g: int,
        dst_b: ArrayLikeFloat,
        dst_stride_b: int,
        *,
        scale: float = 1.0,
        offset: float = 0.0,
        alpha: float = 0.0,
        beta: float = 1.0,
        use_mt: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width/height müssen > 0 sein.")
        if src_stride_bytes < width * 3:
            raise ValueError("src_stride_bytes < width*3 (Bytes) ist ungültig.")
        for stride, name in ((dst_stride_r, "dst_stride_r"), (dst_stride_g, "dst_stride_g"), (dst_stride_b, "dst_stride_b")):
            if stride < width * 4:
                raise ValueError(f"{name} < width*4 (Bytes) ist ungültig.")
        c_src = _to_c_u8_ptr_2d(src, width * 3, height, src_stride_bytes)
        c_r = _to_c_f32_ptr_2d(dst_r, width, height, dst_stride_r)
        c_g = _to_c_f32_ptr_2d(dst_g, width, height, dst_stride_g)
        c_b = _to_c_f32_ptr_2d(dst_b, width, height, dst_stride_b)
        rc = _lib.halo_img_rgb_u8_to_f32_interleaved(
            c_src, C.c_longlong(src_stride_bytes),
            c_r, C.c_longlong(dst_stride_r),
            c_g, C.c_longlong(dst_stride_g),
            c_b, C.c_longlong(dst_stride_b),
            C.c_int(width), C.c_int(height),
            C.c_float(scale), C.c_float(offset),
            C.c_float(alpha), C.c_float(beta),
            C.c_int(1 if use_mt else 0)
        )
        if rc != 0:
            raise RuntimeError(f"halo_img_rgb_u8_to_f32_interleaved() fehlgeschlagen (rc={rc}).")

    def img_rgb_u8_to_f32_planar_2d(
        self,
        src_r: ArrayLikeU8,
        src_g: ArrayLikeU8,
        src_b: ArrayLikeU8,
        width: int,
        height: int,
        src_stride_r: int,
        src_stride_g: int,
        src_stride_b: int,
        dst_r: ArrayLikeFloat,
        dst_stride_r: int,
        dst_g: ArrayLikeFloat,
        dst_stride_g: int,
        dst_b: ArrayLikeFloat,
        dst_stride_b: int,
        *,
        scale: float = 1.0,
        offset: float = 0.0,
        alpha: float = 0.0,
        beta: float = 1.0,
        use_mt: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width/height müssen > 0 sein.")
        for stride, name in ((src_stride_r, "src_stride_r"), (src_stride_g, "src_stride_g"), (src_stride_b, "src_stride_b")):
            if stride < width:
                raise ValueError(f"{name} < width (Bytes) ist ungültig.")
        for stride, name in ((dst_stride_r, "dst_stride_r"), (dst_stride_g, "dst_stride_g"), (dst_stride_b, "dst_stride_b")):
            if stride < width * 4:
                raise ValueError(f"{name} < width*4 (Bytes) ist ungültig.")
        c_r_src = _to_c_u8_ptr_2d(src_r, width, height, src_stride_r)
        c_g_src = _to_c_u8_ptr_2d(src_g, width, height, src_stride_g)
        c_b_src = _to_c_u8_ptr_2d(src_b, width, height, src_stride_b)
        c_r_dst = _to_c_f32_ptr_2d(dst_r, width, height, dst_stride_r)
        c_g_dst = _to_c_f32_ptr_2d(dst_g, width, height, dst_stride_g)
        c_b_dst = _to_c_f32_ptr_2d(dst_b, width, height, dst_stride_b)
        rc = _lib.halo_img_rgb_u8_to_f32_planar(
            c_r_src, C.c_longlong(src_stride_r),
            c_g_src, C.c_longlong(src_stride_g),
            c_b_src, C.c_longlong(src_stride_b),
            c_r_dst, C.c_longlong(dst_stride_r),
            c_g_dst, C.c_longlong(dst_stride_g),
            c_b_dst, C.c_longlong(dst_stride_b),
            C.c_int(width), C.c_int(height),
            C.c_float(scale), C.c_float(offset),
            C.c_float(alpha), C.c_float(beta),
            C.c_int(1 if use_mt else 0)
        )
        if rc != 0:
            raise RuntimeError(f"halo_img_rgb_u8_to_f32_planar() fehlgeschlagen (rc={rc}).")

    def box_blur_f32(
        self,
        src: ArrayLikeFloat,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst: ArrayLikeFloat,
        dst_stride_bytes: int,
        radius: int,
        *,
        use_mt: bool = True,
    ) -> None:
        if src_stride_bytes < width * 4 or dst_stride_bytes < width * 4:
            raise ValueError("stride < width*4 (Bytes) ist ungültig.")
        c_src = _to_c_f32_ptr_2d(src, width, height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, width, height, dst_stride_bytes)
        rc = _lib.halo_img_box_blur_f32(
            c_src, C.c_longlong(src_stride_bytes),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(width), C.c_int(height),
            C.c_int(radius),
            C.c_int(1 if use_mt else 0)
        )
        if rc != 0:
            raise RuntimeError(f"halo_img_box_blur_f32() fehlgeschlagen (rc={rc}).")

    def gauss_blur_f32(
        self,
        src: ArrayLikeFloat,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst: ArrayLikeFloat,
        dst_stride_bytes: int,
        radius: int,
        sigma: float,
        *,
        use_mt: bool = True,
    ) -> None:
        if src_stride_bytes < width * 4 or dst_stride_bytes < width * 4:
            raise ValueError("stride < width*4 (Bytes) ist ungültig.")
        c_src = _to_c_f32_ptr_2d(src, width, height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, width, height, dst_stride_bytes)
        rc = _lib.halo_img_gauss_blur_f32(
            c_src, C.c_longlong(src_stride_bytes),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(width), C.c_int(height),
            C.c_int(radius),
            C.c_float(sigma),
            C.c_int(1 if use_mt else 0)
        )
        if rc != 0:
            raise RuntimeError(f"halo_img_gauss_blur_f32() fehlgeschlagen (rc={rc}).")

    def sobel_f32(
        self,
        src: ArrayLikeFloat,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst_gx: ArrayLikeFloat,
        dst_stride_gx: int,
        dst_gy: ArrayLikeFloat,
        dst_stride_gy: int,
        *,
        use_mt: bool = True,
    ) -> None:
        if src_stride_bytes < width * 4:
            raise ValueError("src_stride_bytes < width*4 (Bytes) ist ungültig.")
        for stride, name in ((dst_stride_gx, "dst_stride_gx"), (dst_stride_gy, "dst_stride_gy")):
            if stride < width * 4:
                raise ValueError(f"{name} < width*4 (Bytes) ist ungültig.")
        c_src = _to_c_f32_ptr_2d(src, width, height, src_stride_bytes)
        c_gx = _to_c_f32_ptr_2d(dst_gx, width, height, dst_stride_gx)
        c_gy = _to_c_f32_ptr_2d(dst_gy, width, height, dst_stride_gy)
        rc = _lib.halo_img_sobel_f32(
            c_src, C.c_longlong(src_stride_bytes),
            c_gx, C.c_longlong(dst_stride_gx),
            c_gy, C.c_longlong(dst_stride_gy),
            C.c_int(width), C.c_int(height),
            C.c_int(1 if use_mt else 0)
        )
        if rc != 0:
            raise RuntimeError(f"halo_img_sobel_f32() fehlgeschlagen (rc={rc}).")

    def resize_f32(
        self,
        src: ArrayLikeFloat,
        src_width: int,
        src_height: int,
        src_stride_bytes: int,
        dst: ArrayLikeFloat,
        dst_width: int,
        dst_height: int,
        dst_stride_bytes: int,
        *,
        filter: Union[str, int] = "bilinear",
        use_mt: bool = True,
    ) -> None:
        if src_stride_bytes < src_width * 4 or dst_stride_bytes < dst_width * 4:
            raise ValueError("stride < width*4 (Bytes) ist ungültig.")
        filter_code: int
        if isinstance(filter, str):
            filt = filter.lower()
            if filt == "bilinear":
                filter_code = 0
            elif filt == "bicubic":
                filter_code = 1
            else:
                raise ValueError("filter muss 'bilinear' oder 'bicubic' sein.")
        else:
            filter_code = int(filter)
        c_src = _to_c_f32_ptr_2d(src, src_width, src_height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, dst_width, dst_height, dst_stride_bytes)
        rc = _lib.halo_img_resize_f32(
            c_src, C.c_longlong(src_stride_bytes),
            C.c_int(src_width), C.c_int(src_height),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(dst_width), C.c_int(dst_height),
            C.c_int(filter_code),
            C.c_int(1 if use_mt else 0)
        )
        if rc != 0:
            raise RuntimeError(f"halo_img_resize_f32() fehlgeschlagen (rc={rc}).")

    def relu_clamp_axpby_f32(
        self,
        src: ArrayLikeFloat,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst: ArrayLikeFloat,
        dst_stride_bytes: int,
        *,
        alpha: float = 0.0,
        beta: float = 1.0,
        clamp_min: float = float('-inf'),
        clamp_max: float = float('inf'),
        apply_relu: bool = False,
        use_mt: bool = True,
    ) -> None:
        if src_stride_bytes < width * 4 or dst_stride_bytes < width * 4:
            raise ValueError("stride < width*4 (Bytes) ist ungültig.")
        c_src = _to_c_f32_ptr_2d(src, width, height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, width, height, dst_stride_bytes)
        rc = _lib.halo_img_relu_clamp_axpby_f32(
            c_src, C.c_longlong(src_stride_bytes),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(width), C.c_int(height),
            C.c_float(alpha), C.c_float(beta),
            C.c_float(clamp_min), C.c_float(clamp_max),
            C.c_int(1 if apply_relu else 0),
            C.c_int(1 if use_mt else 0)
        )
        if rc != 0:
            raise RuntimeError(f"halo_img_relu_clamp_axpby_f32() fehlgeschlagen (rc={rc}).")

    # Optional: manueller Shutdown (z. B. bei langlaufenden Prozessen)
    def close(self) -> None:
        if _HAS_SHUTDOWN:
            try:
                _lib.halo_shutdown_pool()
            except Exception:
                pass

    # ---- intern ----
    def _query_features(self) -> dict:
        sse2 = C.c_int(0); avx2 = C.c_int(0); avx512 = C.c_int(0)
        if _lib.halo_query_features(C.byref(sse2), C.byref(avx2), C.byref(avx512)) != 0:
            raise RuntimeError("halo_query_features() fehlgeschlagen.")
        return {"sse2": bool(sse2.value), "avx2": bool(avx2.value), "avx512": bool(avx512.value)}

# Automatischer Pool-Shutdown beim Interpreter-Exit
def _shutdown_pool_if_available():
    if _HAS_SHUTDOWN:
        try:
            _lib.halo_shutdown_pool()
        except Exception:
            pass

atexit.register(_shutdown_pool_if_available)
