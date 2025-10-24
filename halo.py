# halo.py (v0.5b) — Python 3.12 Wrapper für HALO v0.5b + 2D-Image-Kern
# - NEU: atexit-Hook ruft halo_shutdown_pool(), falls in DLL vorhanden.
# - Optional: HALO.close() für manuellen Shutdown.

from __future__ import annotations
import ctypes as C
import json, math, os, platform, re, atexit
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union, Optional, Any
from array import array


__all__ = [
    "HALO",
    "Impl",
    "make_identity_lut",
    "make_aligned_u8_buffer",
    "make_aligned_f32_buffer",
    "make_pinned_float_array",
    "HALO_VERSION",
    "SVGPath",
    "VectorCanvas",
    "VectorShape",
    "PlotLayout",
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

try:
    _lib.halo_version.restype = C.c_char_p
    HALO_VERSION = _lib.halo_version().decode("utf-8", "ignore")
except Exception:
    HALO_VERSION = "unknown"

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

_lib.halo_box_blur_f32.argtypes = [
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_int,
    C.c_int
]
_lib.halo_box_blur_f32.restype = C.c_int

_lib.halo_gaussian_blur_f32.argtypes = [
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_float,
    C.c_int
]
_lib.halo_gaussian_blur_f32.restype = C.c_int

_lib.halo_sobel_f32.argtypes = [
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_int
]
_lib.halo_sobel_f32.restype = C.c_int

_lib.halo_resize_bilinear_f32.argtypes = [
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_int
]
_lib.halo_resize_bilinear_f32.restype = C.c_int

_lib.halo_resize_bicubic_f32.argtypes = [
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_int
]
_lib.halo_resize_bicubic_f32.restype = C.c_int

_lib.halo_flip_f32.argtypes = [
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_int, C.c_int,
    C.c_int
]
_lib.halo_flip_f32.restype = C.c_int

_lib.halo_rotate90_f32.argtypes = [
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_int,
    C.c_int
]
_lib.halo_rotate90_f32.restype = C.c_int

_lib.halo_invert_f32.argtypes = [
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_float, C.c_float,
    C.c_int,
    C.c_int
]
_lib.halo_invert_f32.restype = C.c_int

_lib.halo_gamma_f32.argtypes = [
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_float,
    C.c_float,
    C.c_int
]
_lib.halo_gamma_f32.restype = C.c_int

_lib.halo_levels_f32.argtypes = [
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_float, C.c_float,
    C.c_float, C.c_float,
    C.c_float,
    C.c_int
]
_lib.halo_levels_f32.restype = C.c_int

_lib.halo_threshold_f32.argtypes = [
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_float, C.c_float,
    C.c_float, C.c_float,
    C.c_int
]
_lib.halo_threshold_f32.restype = C.c_int

_lib.halo_median3x3_f32.argtypes = [
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_int
]
_lib.halo_median3x3_f32.restype = C.c_int

_lib.halo_erode3x3_f32.argtypes = [
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_int
]
_lib.halo_erode3x3_f32.restype = C.c_int

_lib.halo_dilate3x3_f32.argtypes = [
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_int
]
_lib.halo_dilate3x3_f32.restype = C.c_int

_lib.halo_open3x3_f32.argtypes = [
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_int
]
_lib.halo_open3x3_f32.restype = C.c_int

_lib.halo_close3x3_f32.argtypes = [
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_int
]
_lib.halo_close3x3_f32.restype = C.c_int

_lib.halo_unsharp_mask_f32.argtypes = [
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_float,
    C.c_float,
    C.c_float,
    C.c_int
]
_lib.halo_unsharp_mask_f32.restype = C.c_int

_lib.halo_relu_clamp_axpby_f32.argtypes = [
    C.POINTER(C.c_float), C.c_longlong,
    C.POINTER(C.c_float), C.c_longlong,
    C.c_int, C.c_int,
    C.c_float, C.c_float,
    C.c_float, C.c_float,
    C.c_int,
    C.c_int
]
_lib.halo_relu_clamp_axpby_f32.restype = C.c_int

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
ArrayLikeU16   = Union[array, memoryview, bytes, bytearray]


# ------------------------------------------------------------
#  Vektorgrafik: Parser, Rasterizer, Plot-Layout
# ------------------------------------------------------------


def _ensure_mutable_f32_view(buf: ArrayLikeFloat) -> memoryview:
    if isinstance(buf, array) and buf.typecode == 'f':
        return memoryview(buf)
    if isinstance(buf, memoryview) and buf.format == 'f' and buf.contiguous and not buf.readonly:
        return buf
    raise TypeError("Erwarte array('f') oder beschreibbaren memoryview('f').")

def _distance_point_segment(px: float, py: float, x0: float, y0: float, x1: float, y1: float) -> float:
    dx = x1 - x0
    dy = y1 - y0
    if dx == 0.0 and dy == 0.0:
        return math.hypot(px - x0, py - y0)
    t = ((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy)
    if t <= 0.0:
        return math.hypot(px - x0, py - y0)
    if t >= 1.0:
        return math.hypot(px - x1, py - y1)
    cx = x0 + t * dx
    cy = y0 + t * dy
    return math.hypot(px - cx, py - cy)


def _point_in_polygon(x: float, y: float, pts: Sequence[Tuple[float, float]]) -> bool:
    inside = False
    n = len(pts)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = pts[i]
        xj, yj = pts[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _flatten_quadratic(p0, p1, p2, tolerance: float) -> List[Tuple[float, float]]:
    def _approx(a, b, c, out):
        ax, ay = a
        bx, by = b
        cx, cy = c
        midx = 0.25 * (ax + 2 * bx + cx)
        midy = 0.25 * (ay + 2 * by + cy)
        chord_mid_x = 0.5 * (ax + cx)
        chord_mid_y = 0.5 * (ay + cy)
        if math.hypot(midx - chord_mid_x, midy - chord_mid_y) <= tolerance:
            out.append(c)
        else:
            ab = ((ax + bx) * 0.5, (ay + by) * 0.5)
            bc = ((bx + cx) * 0.5, (by + cy) * 0.5)
            abc = ((ab[0] + bc[0]) * 0.5, (ab[1] + bc[1]) * 0.5)
            _approx(a, ab, abc, out)
            _approx(abc, bc, c, out)

    result = [p0]
    _approx(p0, p1, p2, result)
    return result


def _flatten_cubic(p0, p1, p2, p3, tolerance: float) -> List[Tuple[float, float]]:
    def _approx(a, b, c, d, out):
        ax, ay = a
        bx, by = b
        cx, cy = c
        dx, dy = d
        chord_mid1 = ((ax + dx) * 0.5, (ay + dy) * 0.5)
        hull_mid1 = ((ax + 3 * (bx + cx) + dx) * 0.125, (ay + 3 * (by + cy) + dy) * 0.125)
        if math.hypot(chord_mid1[0] - hull_mid1[0], chord_mid1[1] - hull_mid1[1]) <= tolerance:
            out.append(d)
        else:
            ab = ((ax + bx) * 0.5, (ay + by) * 0.5)
            bc = ((bx + cx) * 0.5, (by + cy) * 0.5)
            cd = ((cx + dx) * 0.5, (cy + dy) * 0.5)
            abc = ((ab[0] + bc[0]) * 0.5, (ab[1] + bc[1]) * 0.5)
            bcd = ((bc[0] + cd[0]) * 0.5, (bc[1] + cd[1]) * 0.5)
            abcd = ((abc[0] + bcd[0]) * 0.5, (abc[1] + bcd[1]) * 0.5)
            _approx(a, ab, abc, abcd, out)
            _approx(abcd, bcd, cd, d, out)

    result = [p0]
    _approx(p0, p1, p2, p3, result)
    return result


@dataclass
class PathCommand:
    command: str
    points: List[Tuple[float, float]]


class SVGPath:
    """Einfacher Parser für SVG-Pfad-Daten (Untermenge: M, L, H, V, C, Q, Z)."""

    _token_re = re.compile(r"[MmLlHhVvCcQqZz]|[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    def __init__(self, commands: List[PathCommand]):
        self.commands = commands

    @classmethod
    def parse(cls, d: str) -> "SVGPath":
        tokens = cls._token_re.findall(d)
        if not tokens:
            return cls([])
        i = 0
        cmd = None
        x = y = 0.0
        start_x = start_y = 0.0
        commands: List[PathCommand] = []

        def take_numbers(count: int) -> List[float]:
            nonlocal i
            values = []
            for _ in range(count):
                if i >= len(tokens):
                    raise ValueError("SVG-Pfad: unvollständige Koordinaten")
                values.append(float(tokens[i]))
                i += 1
            return values

        while i < len(tokens):
            token = tokens[i]
            if re.match(r"[A-Za-z]", token):
                cmd = token
                i += 1
            if cmd is None:
                raise ValueError("SVG-Pfad: Kommando erwartet")
            if cmd in "Zz":
                commands.append(PathCommand("Z", []))
                x, y = start_x, start_y
                cmd = cmd  # keep last command for potential reuse
                continue
            if cmd in "Mm":
                nums = []
                while i < len(tokens) and not re.match(r"[A-Za-z]", tokens[i]):
                    nums.append(float(tokens[i]))
                    i += 1
                if len(nums) % 2 != 0:
                    raise ValueError("SVG-Pfad: M benötigt Paare")
                for idx in range(0, len(nums), 2):
                    nx = nums[idx]
                    ny = nums[idx + 1]
                    if cmd == "m":
                        nx += x
                        ny += y
                    x, y = nx, ny
                    if idx == 0:
                        start_x, start_y = x, y
                        commands.append(PathCommand("M", [(x, y)]))
                    else:
                        commands.append(PathCommand("L", [(x, y)]))
                cmd = "L" if cmd == "M" else "l"
                continue
            if cmd in "Ll":
                while i < len(tokens) and not re.match(r"[A-Za-z]", tokens[i]):
                    nx = float(tokens[i]); ny = float(tokens[i + 1])
                    i += 2
                    if cmd == "l":
                        nx += x
                        ny += y
                    x, y = nx, ny
                    commands.append(PathCommand("L", [(x, y)]))
                continue
            if cmd in "Hh":
                while i < len(tokens) and not re.match(r"[A-Za-z]", tokens[i]):
                    nx = float(tokens[i]); i += 1
                    if cmd == "h":
                        nx += x
                    x = nx
                    commands.append(PathCommand("L", [(x, y)]))
                continue
            if cmd in "Vv":
                while i < len(tokens) and not re.match(r"[A-Za-z]", tokens[i]):
                    ny = float(tokens[i]); i += 1
                    if cmd == "v":
                        ny += y
                    y = ny
                    commands.append(PathCommand("L", [(x, y)]))
                continue
            if cmd in "Qq":
                while i < len(tokens) and not re.match(r"[A-Za-z]", tokens[i]):
                    vals = take_numbers(4)
                    cx, cy, nx, ny = vals
                    if cmd == "q":
                        cx += x; cy += y; nx += x; ny += y
                    commands.append(PathCommand("Q", [(cx, cy), (nx, ny)]))
                    x, y = nx, ny
                continue
            if cmd in "Cc":
                while i < len(tokens) and not re.match(r"[A-Za-z]", tokens[i]):
                    vals = take_numbers(6)
                    c1x, c1y, c2x, c2y, nx, ny = vals
                    if cmd == "c":
                        c1x += x; c1y += y; c2x += x; c2y += y; nx += x; ny += y
                    commands.append(PathCommand("C", [(c1x, c1y), (c2x, c2y), (nx, ny)]))
                    x, y = nx, ny
                continue
            raise ValueError(f"SVG-Pfad: unbekanntes Kommando {cmd}")

        return cls(commands)

    def to_polylines(self, tolerance: float = 0.5) -> Tuple[List[List[Tuple[float, float]]], List[bool]]:
        contours: List[List[Tuple[float, float]]] = []
        closed: List[bool] = []
        current: List[Tuple[float, float]] = []
        x = y = 0.0
        start: Optional[Tuple[float, float]] = None
        for cmd in self.commands:
            if cmd.command == "M":
                if current:
                    contours.append(current)
                    closed.append(False)
                current = [cmd.points[0]]
                x, y = cmd.points[0]
                start = (x, y)
                continue
            if cmd.command == "L":
                if not current:
                    current = [(x, y)]
                current.append(cmd.points[0])
                x, y = cmd.points[0]
                continue
            if cmd.command == "Q":
                if not current:
                    current = [(x, y)]
                pts = _flatten_quadratic((x, y), cmd.points[0], cmd.points[1], tolerance)
                current.extend(pts[1:])
                x, y = cmd.points[1]
                continue
            if cmd.command == "C":
                if not current:
                    current = [(x, y)]
                pts = _flatten_cubic((x, y), cmd.points[0], cmd.points[1], cmd.points[2], tolerance)
                current.extend(pts[1:])
                x, y = cmd.points[2]
                continue
            if cmd.command == "Z":
                if current:
                    if start is not None and current[-1] != start:
                        current.append(start)
                    contours.append(current)
                    closed.append(True)
                    current = []
                    if start is not None:
                        x, y = start
                    start = None
                continue
        if current:
            contours.append(current)
            closed.append(False)
        return contours, closed


@dataclass
class VectorShape:
    contours: List[List[Tuple[float, float]]]
    closed: List[bool]
    fill_color: Optional[float] = 1.0
    stroke_color: Optional[float] = 1.0
    stroke_width: float = 1.0

    @classmethod
    def from_svg(cls, d: str, *, tolerance: float = 0.5,
                 fill_color: Optional[float] = 1.0,
                 stroke_color: Optional[float] = 1.0,
                 stroke_width: float = 1.0) -> "VectorShape":
        path = SVGPath.parse(d)
        contours, closed = path.to_polylines(tolerance)
        return cls(contours, closed, fill_color=fill_color,
                   stroke_color=stroke_color, stroke_width=stroke_width)


DEFAULT_FONT: dict[str, Tuple[int, int, List[str]]] = {
    "0": (5, 7, [
        " ### ",
        "#   #",
        "#  ##",
        "# # #",
        "##  #",
        "#   #",
        " ### ",
    ]),
    "1": (5, 7, [
        "  #  ",
        " ##  ",
        "  #  ",
        "  #  ",
        "  #  ",
        "  #  ",
        " ### ",
    ]),
    "2": (5, 7, [
        " ### ",
        "#   #",
        "    #",
        "   # ",
        "  #  ",
        " #   ",
        "#####",
    ]),
    "3": (5, 7, [
        " ### ",
        "#   #",
        "    #",
        "  ## ",
        "    #",
        "#   #",
        " ### ",
    ]),
    "4": (5, 7, [
        "   # ",
        "  ## ",
        " # # ",
        "#  # ",
        "#####",
        "   # ",
        "   # ",
    ]),
    "5": (5, 7, [
        "#####",
        "#    ",
        "#    ",
        "#### ",
        "    #",
        "#   #",
        " ### ",
    ]),
    "6": (5, 7, [
        " ### ",
        "#   #",
        "#    ",
        "#### ",
        "#   #",
        "#   #",
        " ### ",
    ]),
    "7": (5, 7, [
        "#####",
        "    #",
        "   # ",
        "  #  ",
        " #   ",
        " #   ",
        " #   ",
    ]),
    "8": (5, 7, [
        " ### ",
        "#   #",
        "#   #",
        " ### ",
        "#   #",
        "#   #",
        " ### ",
    ]),
    "9": (5, 7, [
        " ### ",
        "#   #",
        "#   #",
        " ####",
        "    #",
        "#   #",
        " ### ",
    ]),
}

DEFAULT_FONT["?"] = (5, 7, [
    " ### ",
    "#   #",
    "    #",
    "   # ",
    "  #  ",
    "     ",
    "  #  ",
])

for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    if ch not in DEFAULT_FONT:
        pattern = [
            "#####",
            "#   #",
            "#   #",
            "#####",
            "#   #",
            "#   #",
            "#   #",
        ]
        DEFAULT_FONT[ch] = (5, 7, pattern)
for ch in "abcdefghijklmnopqrstuvwxyz":
    if ch not in DEFAULT_FONT:
        DEFAULT_FONT[ch] = DEFAULT_FONT[ch.upper()]
DEFAULT_FONT[" "] = (3, 7, ["   "] * 7)
DEFAULT_FONT["-"] = (3, 7, ["   ", "   ", "   ", "###", "   ", "   ", "   "])
DEFAULT_FONT["."] = (1, 7, [" ", " ", " ", " ", " ", "##", "##"])
DEFAULT_FONT[","] = (2, 7, ["  ", "  ", "  ", "  ", "##", "##", " #"])


class VectorCanvas:
    """Einfache Float-Canvas mit antialiasenden Zeichenfunktionen."""

    def __init__(self, width: int, height: int, background: float = 0.0):
        if width <= 0 or height <= 0:
            raise ValueError("Canvas muss positive Abmessungen besitzen")
        self.width = int(width)
        self.height = int(height)
        self.buffer = array('f', [float(background)] * (self.width * self.height))

    def clear(self, value: float = 0.0) -> None:
        for i in range(len(self.buffer)):
            self.buffer[i] = value

    def blend_pixel(self, x: int, y: int, value: float, alpha: float) -> None:
        if alpha <= 0.0:
            return
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = y * self.width + x
            base = self.buffer[idx]
            self.buffer[idx] = base * (1.0 - alpha) + value * alpha

    def stroke_segment(self, p0: Tuple[float, float], p1: Tuple[float, float],
                       color: float, width: float) -> None:
        x0, y0 = p0
        x1, y1 = p1
        half = max(width, 0.01) * 0.5
        min_x = max(int(math.floor(min(x0, x1) - half - 1)), 0)
        max_x = min(int(math.ceil(max(x0, x1) + half + 1)), self.width - 1)
        min_y = max(int(math.floor(min(y0, y1) - half - 1)), 0)
        max_y = min(int(math.ceil(max(y0, y1) + half + 1)), self.height - 1)
        for py in range(min_y, max_y + 1):
            for px in range(min_x, max_x + 1):
                dist = _distance_point_segment(px + 0.5, py + 0.5, x0, y0, x1, y1)
                if dist <= half + 1.0:
                    if half <= 0.5:
                        alpha = max(0.0, 1.0 - dist)
                    else:
                        alpha = max(0.0, min(1.0, 1.0 - (dist - half) / max(half, 1e-6)))
                    if alpha > 0.0:
                        self.blend_pixel(px, py, color, min(alpha, 1.0))

    def stroke_polyline(self, points: Sequence[Tuple[float, float]], color: float,
                        width: float = 1.0, closed: bool = False) -> None:
        if len(points) < 2:
            return
        for i in range(len(points) - 1):
            self.stroke_segment(points[i], points[i + 1], color, width)
        if closed:
            self.stroke_segment(points[-1], points[0], color, width)

    def fill_polygon(self, points: Sequence[Tuple[float, float]], color: float,
                     samples: int = 4) -> None:
        if len(points) < 3:
            return
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x = max(int(math.floor(min(xs))), 0)
        max_x = min(int(math.ceil(max(xs))), self.width)
        min_y = max(int(math.floor(min(ys))), 0)
        max_y = min(int(math.ceil(max(ys))), self.height)
        step = 1.0 / samples
        weight = step * step
        for py in range(min_y, max_y):
            for px in range(min_x, max_x):
                acc = 0.0
                for sy in range(samples):
                    for sx in range(samples):
                        sxp = px + (sx + 0.5) * step
                        syp = py + (sy + 0.5) * step
                        if _point_in_polygon(sxp, syp, points):
                            acc += weight
                if acc > 0.0:
                    self.blend_pixel(px, py, color, min(1.0, acc))

    def draw_text(self, x: float, y: float, text: str, size: float = 12.0,
                  color: float = 1.0, baseline: str = "alphabetic") -> None:
        scale = max(size / 7.0, 0.01)
        cursor_x = x
        cursor_y = y
        if baseline == "alphabetic":
            cursor_y -= size
        for ch in text:
            glyph = DEFAULT_FONT.get(ch)
            if glyph is None:
                glyph = DEFAULT_FONT.get("?", DEFAULT_FONT[" "])
            gw, gh, rows = glyph
            for gy, row in enumerate(rows):
                for gx, c in enumerate(row):
                    if c != "#":
                        continue
                    px = cursor_x + gx * scale
                    py = cursor_y + gy * scale
                    self.fill_polygon([
                        (px, py),
                        (px + scale, py),
                        (px + scale, py + scale),
                        (px, py + scale),
                    ], color, samples=1)
            cursor_x += (gw + 1) * scale

    def draw_shape(self, shape: "VectorShape") -> None:
        for pts, is_closed in zip(shape.contours, shape.closed):
            if shape.fill_color is not None and is_closed:
                self.fill_polygon(pts, float(shape.fill_color))
        if shape.stroke_color is not None and shape.stroke_width > 0.0:
            for pts, is_closed in zip(shape.contours, shape.closed):
                self.stroke_polyline(pts, float(shape.stroke_color),
                                     width=shape.stroke_width, closed=is_closed)

    def blit_to(self, dst: ArrayLikeFloat, dst_stride_bytes: int) -> None:
        mv = _ensure_mutable_f32_view(dst)
        row_bytes = self.width * 4
        if dst_stride_bytes < row_bytes:
            raise ValueError("dst_stride_bytes zu klein für Canvas-Breite")
        if dst_stride_bytes % 4 != 0:
            raise ValueError("dst_stride_bytes muss Vielfaches von 4 sein")
        pitch = dst_stride_bytes // 4
        if mv.nbytes < dst_stride_bytes * self.height:
            raise ValueError("dst-Puffer zu klein für Canvas-Höhe")
        for y in range(self.height):
            start = y * self.width
            end = start + self.width
            dst_offset = pitch * y
            mv[dst_offset:dst_offset + self.width] = self.buffer[start:end]


class PlotLayout:
    """Hilfsklasse für einfache Diagramm-Layouts auf einem VectorCanvas."""

    def __init__(self, canvas: VectorCanvas, padding: int = 32):
        self.canvas = canvas
        self.padding = padding

    def _axis_bounds(self) -> Tuple[int, int, int, int]:
        return (self.padding, self.padding,
                self.canvas.width - self.padding,
                self.canvas.height - self.padding)

    def draw_axes(self, x_label: str = "", y_label: str = "", title: str = "",
                  tick_count: int = 5) -> None:
        x0, y0, x1, y1 = self._axis_bounds()
        self.canvas.stroke_segment((x0, y1), (x1, y1), 1.0, 1.5)
        self.canvas.stroke_segment((x0, y0), (x0, y1), 1.0, 1.5)
        if title:
            self.canvas.draw_text(x0, self.padding * 0.5, title, size=14, color=1.0, baseline="alphabetic")
        if x_label:
            self.canvas.draw_text((x0 + x1) * 0.5 - len(x_label) * 3, self.canvas.height - self.padding * 0.4,
                                  x_label, size=12, color=1.0, baseline="alphabetic")
        if y_label:
            self.canvas.draw_text(self.padding * 0.1, (y0 + y1) * 0.5, y_label, size=12, color=1.0, baseline="alphabetic")
        for i in range(tick_count + 1):
            t = i / tick_count if tick_count else 0
            tx = x0 + t * (x1 - x0)
            ty = y1 - t * (y1 - y0)
            self.canvas.stroke_segment((tx, y1), (tx, y1 + 5), 1.0, 1.0)
            self.canvas.stroke_segment((x0 - 5, ty), (x0, ty), 1.0, 1.0)

    def plot_line(self, xs: Sequence[float], ys: Sequence[float], color: float = 1.0,
                  *, tick_count: int = 5, x_label: str = "", y_label: str = "",
                  title: str = "") -> None:
        if len(xs) != len(ys):
            raise ValueError("xs und ys müssen gleiche Länge haben")
        if not xs:
            return
        x0, y0, x1, y1 = self._axis_bounds()
        min_x = min(xs); max_x = max(xs)
        min_y = min(ys); max_y = max(ys)
        span_x = max(max_x - min_x, 1e-5)
        span_y = max(max_y - min_y, 1e-5)
        points = []
        for vx, vy in zip(xs, ys):
            sx = x0 + (vx - min_x) / span_x * (x1 - x0)
            sy = y1 - (vy - min_y) / span_y * (y1 - y0)
            points.append((sx, sy))
        self.canvas.stroke_polyline(points, color, width=1.5)
        self.draw_axes(x_label=x_label, y_label=y_label, title=title, tick_count=tick_count)


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

def _to_c_f32_ptr_2d(buf: ArrayLikeFloat, width: int, height: int, stride_bytes: int, *, components: int = 1) -> C.POINTER(C.c_float):
    match buf:
        case arr if isinstance(arr, array) and arr.typecode == 'f':
            mv = memoryview(arr)
        case mv if isinstance(mv, memoryview) and mv.format == 'f' and mv.contiguous and not mv.readonly:
            pass
        case _:
            raise TypeError("dst erwartet array('f') oder schreibbaren, kontiguösen memoryview('f').")
    if components <= 0:
        raise ValueError("components muss >= 1 sein.")
    min_bytes = (height - 1) * stride_bytes + width * components * 4
    if mv.nbytes < min_bytes:
        raise ValueError(f"dst: zu klein für height={height}, stride={stride_bytes}, width={width}, components={components}.")
    c_arr = (C.c_float * (mv.nbytes // 4)).from_buffer(mv)
    return C.cast(c_arr, C.POINTER(C.c_float))

def _to_c_u16_ptr_2d(buf: ArrayLikeU16, width: int, height: int, stride_bytes: int) -> C.POINTER(C.c_ushort):
    if isinstance(buf, array) and buf.typecode == 'H':
        mv = memoryview(buf)
    elif isinstance(buf, (bytes, bytearray)):
        mv = memoryview(buf).cast('H')
    elif isinstance(buf, memoryview):
        mv = buf
        if mv.format != 'H':
            mv = mv.cast('H')
    else:
        raise TypeError("src erwartet array('H'), memoryview('H') oder bytes/bytearray.")
    if mv.itemsize != 2:
        mv = mv.cast('H')
    if not mv.contiguous:
        raise TypeError("src muss kontiguös sein.")
    min_bytes = (height - 1) * stride_bytes + width * 2
    if mv.nbytes < min_bytes:
        raise ValueError(f"src: zu klein für height={height}, stride={stride_bytes}, width={width}.")
    count = mv.nbytes // 2
    if mv.readonly:
        c_arr = (C.c_ushort * count).from_buffer_copy(mv)
    else:
        c_arr = (C.c_ushort * count).from_buffer(mv)
    return C.cast(c_arr, C.POINTER(C.c_ushort))

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

class AlignedBuffer:
    __slots__ = ("_raw", "_offset", "size", "alignment")

    def __init__(self, size: int, alignment: int):
        if alignment <= 0:
            raise ValueError("alignment muss > 0 sein.")
        buf_type = C.c_ubyte * (size + alignment)
        raw = buf_type()
        addr = C.addressof(raw)
        offset = (alignment - (addr % alignment)) % alignment
        self._raw = raw
        self._offset = offset
        self.size = size
        self.alignment = alignment

    def view(self, fmt: str = 'B') -> memoryview:
        base = memoryview(self._raw).cast('B')
        mv = base[self._offset:self._offset + self.size]
        return mv.cast(fmt) if fmt != 'B' else mv

def make_aligned_u8_buffer(width: int, height: int, *, alignment: int = 64, stride_bytes: int | None = None) -> tuple[memoryview, int]:
    if width <= 0 or height <= 0:
        raise ValueError("width/height müssen > 0 sein.")
    if stride_bytes is None:
        stride_bytes = ((width + alignment - 1) // alignment) * alignment
    if stride_bytes < width:
        raise ValueError("stride_bytes zu klein.")
    buf = AlignedBuffer(stride_bytes * height, alignment)
    return buf.view('B'), stride_bytes

def make_aligned_f32_buffer(
    width: int,
    height: int,
    *,
    components: int = 1,
    alignment: int = 64,
    stride_bytes: int | None = None,
) -> tuple[memoryview, int]:
    if width <= 0 or height <= 0:
        raise ValueError("width/height müssen > 0 sein.")
    if components <= 0:
        raise ValueError("components müssen > 0 sein.")
    row_bytes = width * components * 4
    if stride_bytes is None:
        stride_bytes = ((row_bytes + alignment - 1) // alignment) * alignment
    if stride_bytes < row_bytes:
        raise ValueError("stride_bytes zu klein.")
    buf = AlignedBuffer(stride_bytes * height, alignment)
    return buf.view('f'), stride_bytes

def make_pinned_float_array(length: int, *, alignment: int = 64) -> memoryview:
    if length <= 0:
        raise ValueError("length muss > 0 sein.")
    buf = AlignedBuffer(length * 4, alignment)
    return buf.view('f')

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
        bit_depth: int,
        *,
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
            C.c_int(bit_depth),
            C.c_float(scale), C.c_float(offset),
            C.c_float(alpha), C.c_float(beta),
            C.c_int(1 if use_mt else 0)
        )
        if rc != 0:
            raise RuntimeError(f"halo_img_u16_to_f32_axpby() fehlgeschlagen (rc={rc}).")

    def img_rgb_u8_to_f32_interleaved(
        self,
        src: ArrayLikeU8,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst: ArrayLikeFloat,
        dst_stride_bytes: int,
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
        if dst_stride_bytes < width * 3 * 4:
            raise ValueError("dst_stride_bytes < width*12 (Bytes) ist ungültig.")
        c_src = _to_c_u8_ptr_2d(src, width * 3, height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, width, height, dst_stride_bytes, components=3)
        rc = _lib.halo_img_rgb_u8_to_f32_interleaved(
            c_src, C.c_longlong(src_stride_bytes),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(width), C.c_int(height),
            C.c_float(scale), C.c_float(offset),
            C.c_float(alpha), C.c_float(beta),
            C.c_int(1 if use_mt else 0)
        )
        if rc != 0:
            raise RuntimeError(f"halo_img_rgb_u8_to_f32_interleaved() fehlgeschlagen (rc={rc}).")

    def img_rgb_u8_to_f32_planar(
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
        dst_g: ArrayLikeFloat,
        dst_b: ArrayLikeFloat,
        dst_stride_r: int,
        dst_stride_g: int,
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
        if src_stride_r < width or src_stride_g < width or src_stride_b < width:
            raise ValueError("src_stride (planar) zu klein.")
        if dst_stride_r < width * 4 or dst_stride_g < width * 4 or dst_stride_b < width * 4:
            raise ValueError("dst_stride (planar) zu klein.")
        c_sr = _to_c_u8_ptr_2d(src_r, width, height, src_stride_r)
        c_sg = _to_c_u8_ptr_2d(src_g, width, height, src_stride_g)
        c_sb = _to_c_u8_ptr_2d(src_b, width, height, src_stride_b)
        c_dr = _to_c_f32_ptr_2d(dst_r, width, height, dst_stride_r)
        c_dg = _to_c_f32_ptr_2d(dst_g, width, height, dst_stride_g)
        c_db = _to_c_f32_ptr_2d(dst_b, width, height, dst_stride_b)
        rc = _lib.halo_img_rgb_u8_to_f32_planar(
            c_sr, C.c_longlong(src_stride_r),
            c_sg, C.c_longlong(src_stride_g),
            c_sb, C.c_longlong(src_stride_b),
            c_dr, C.c_longlong(dst_stride_r),
            c_dg, C.c_longlong(dst_stride_g),
            c_db, C.c_longlong(dst_stride_b),
            C.c_int(width), C.c_int(height),
            C.c_float(scale), C.c_float(offset),
            C.c_float(alpha), C.c_float(beta),
            C.c_int(1 if use_mt else 0)
        )
        if rc != 0:
            raise RuntimeError(f"halo_img_rgb_u8_to_f32_planar() fehlgeschlagen (rc={rc}).")

    def flip_f32(
        self,
        src: ArrayLikeFloat,
        dst: ArrayLikeFloat,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst_stride_bytes: int,
        *,
        horizontal: bool = False,
        vertical: bool = True,
        use_mt: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width/height müssen > 0 sein.")
        if src_stride_bytes < width * 4:
            raise ValueError("src_stride_bytes zu klein für float32-Bild.")
        if dst_stride_bytes < width * 4:
            raise ValueError("dst_stride_bytes zu klein für float32-Bild.")
        c_src = _to_c_f32_ptr_2d(src, width, height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, width, height, dst_stride_bytes)
        rc = _lib.halo_flip_f32(
            c_src, C.c_longlong(src_stride_bytes),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(width), C.c_int(height),
            C.c_int(1 if horizontal else 0),
            C.c_int(1 if vertical else 0),
            C.c_int(1 if use_mt else 0),
        )
        if rc != 0:
            raise RuntimeError(f"halo_flip_f32() fehlgeschlagen (rc={rc}).")

    def rotate90_f32(
        self,
        src: ArrayLikeFloat,
        dst: ArrayLikeFloat,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst_stride_bytes: int,
        *,
        quarter_turns: int,
        use_mt: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width/height müssen > 0 sein.")
        dst_width = height if quarter_turns % 2 else width
        dst_height = width if quarter_turns % 2 else height
        if src_stride_bytes < width * 4:
            raise ValueError("src_stride_bytes zu klein für float32-Bild.")
        if dst_stride_bytes < dst_width * 4:
            raise ValueError("dst_stride_bytes zu klein für Zielbild.")
        c_src = _to_c_f32_ptr_2d(src, width, height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, dst_width, dst_height, dst_stride_bytes)
        rc = _lib.halo_rotate90_f32(
            c_src, C.c_longlong(src_stride_bytes),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(width), C.c_int(height),
            C.c_int(quarter_turns),
            C.c_int(1 if use_mt else 0),
        )
        if rc == -5:
            raise ValueError("In-place-Rotation um 90°/270° erfordert separaten Zielpuffer.")
        if rc != 0:
            raise RuntimeError(f"halo_rotate90_f32() fehlgeschlagen (rc={rc}).")

    def invert_f32(
        self,
        src: ArrayLikeFloat,
        dst: ArrayLikeFloat,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst_stride_bytes: int,
        *,
        min_val: float = 0.0,
        max_val: float = 1.0,
        use_range: bool = True,
        use_mt: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width/height müssen > 0 sein.")
        if src_stride_bytes < width * 4 or dst_stride_bytes < width * 4:
            raise ValueError("Stride zu klein für float32-Bild.")
        if use_range and not (max_val > min_val):
            raise ValueError("max_val muss größer als min_val sein.")
        c_src = _to_c_f32_ptr_2d(src, width, height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, width, height, dst_stride_bytes)
        rc = _lib.halo_invert_f32(
            c_src, C.c_longlong(src_stride_bytes),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(width), C.c_int(height),
            C.c_float(min_val), C.c_float(max_val),
            C.c_int(1 if use_range else 0),
            C.c_int(1 if use_mt else 0),
        )
        if rc != 0:
            raise RuntimeError(f"halo_invert_f32() fehlgeschlagen (rc={rc}).")

    def gamma_f32(
        self,
        src: ArrayLikeFloat,
        dst: ArrayLikeFloat,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst_stride_bytes: int,
        *,
        gamma: float,
        gain: float = 1.0,
        use_mt: bool = True,
    ) -> None:
        if gamma <= 0.0:
            raise ValueError("gamma muss > 0 sein.")
        if width <= 0 or height <= 0:
            raise ValueError("width/height müssen > 0 sein.")
        if src_stride_bytes < width * 4 or dst_stride_bytes < width * 4:
            raise ValueError("Stride zu klein für float32-Bild.")
        c_src = _to_c_f32_ptr_2d(src, width, height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, width, height, dst_stride_bytes)
        rc = _lib.halo_gamma_f32(
            c_src, C.c_longlong(src_stride_bytes),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(width), C.c_int(height),
            C.c_float(gamma), C.c_float(gain),
            C.c_int(1 if use_mt else 0),
        )
        if rc != 0:
            raise RuntimeError(f"halo_gamma_f32() fehlgeschlagen (rc={rc}).")

    def levels_f32(
        self,
        src: ArrayLikeFloat,
        dst: ArrayLikeFloat,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst_stride_bytes: int,
        *,
        in_low: float,
        in_high: float,
        out_low: float = 0.0,
        out_high: float = 1.0,
        gamma: float = 1.0,
        use_mt: bool = True,
    ) -> None:
        if in_high <= in_low:
            raise ValueError("in_high muss größer als in_low sein.")
        if width <= 0 or height <= 0:
            raise ValueError("width/height müssen > 0 sein.")
        if src_stride_bytes < width * 4 or dst_stride_bytes < width * 4:
            raise ValueError("Stride zu klein für float32-Bild.")
        if gamma <= 0.0:
            raise ValueError("gamma muss > 0 sein.")
        c_src = _to_c_f32_ptr_2d(src, width, height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, width, height, dst_stride_bytes)
        rc = _lib.halo_levels_f32(
            c_src, C.c_longlong(src_stride_bytes),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(width), C.c_int(height),
            C.c_float(in_low), C.c_float(in_high),
            C.c_float(out_low), C.c_float(out_high),
            C.c_float(gamma),
            C.c_int(1 if use_mt else 0),
        )
        if rc != 0:
            raise RuntimeError(f"halo_levels_f32() fehlgeschlagen (rc={rc}).")

    def threshold_f32(
        self,
        src: ArrayLikeFloat,
        dst: ArrayLikeFloat,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst_stride_bytes: int,
        *,
        low: float,
        high: float,
        low_value: float = 0.0,
        high_value: float = 1.0,
        use_mt: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width/height müssen > 0 sein.")
        if src_stride_bytes < width * 4 or dst_stride_bytes < width * 4:
            raise ValueError("Stride zu klein für float32-Bild.")
        c_src = _to_c_f32_ptr_2d(src, width, height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, width, height, dst_stride_bytes)
        rc = _lib.halo_threshold_f32(
            c_src, C.c_longlong(src_stride_bytes),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(width), C.c_int(height),
            C.c_float(low), C.c_float(high),
            C.c_float(low_value), C.c_float(high_value),
            C.c_int(1 if use_mt else 0),
        )
        if rc != 0:
            raise RuntimeError(f"halo_threshold_f32() fehlgeschlagen (rc={rc}).")

    def median3x3_f32(
        self,
        src: ArrayLikeFloat,
        dst: ArrayLikeFloat,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst_stride_bytes: int,
        *,
        use_mt: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width/height müssen > 0 sein.")
        if src_stride_bytes < width * 4 or dst_stride_bytes < width * 4:
            raise ValueError("Stride zu klein für float32-Bild.")
        c_src = _to_c_f32_ptr_2d(src, width, height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, width, height, dst_stride_bytes)
        rc = _lib.halo_median3x3_f32(
            c_src, C.c_longlong(src_stride_bytes),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(width), C.c_int(height),
            C.c_int(1 if use_mt else 0),
        )
        if rc != 0:
            raise RuntimeError(f"halo_median3x3_f32() fehlgeschlagen (rc={rc}).")

    def erode3x3_f32(
        self,
        src: ArrayLikeFloat,
        dst: ArrayLikeFloat,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst_stride_bytes: int,
        *,
        use_mt: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width/height müssen > 0 sein.")
        if src_stride_bytes < width * 4 or dst_stride_bytes < width * 4:
            raise ValueError("Stride zu klein für float32-Bild.")
        c_src = _to_c_f32_ptr_2d(src, width, height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, width, height, dst_stride_bytes)
        rc = _lib.halo_erode3x3_f32(
            c_src, C.c_longlong(src_stride_bytes),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(width), C.c_int(height),
            C.c_int(1 if use_mt else 0),
        )
        if rc != 0:
            raise RuntimeError(f"halo_erode3x3_f32() fehlgeschlagen (rc={rc}).")

    def dilate3x3_f32(
        self,
        src: ArrayLikeFloat,
        dst: ArrayLikeFloat,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst_stride_bytes: int,
        *,
        use_mt: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width/height müssen > 0 sein.")
        if src_stride_bytes < width * 4 or dst_stride_bytes < width * 4:
            raise ValueError("Stride zu klein für float32-Bild.")
        c_src = _to_c_f32_ptr_2d(src, width, height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, width, height, dst_stride_bytes)
        rc = _lib.halo_dilate3x3_f32(
            c_src, C.c_longlong(src_stride_bytes),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(width), C.c_int(height),
            C.c_int(1 if use_mt else 0),
        )
        if rc != 0:
            raise RuntimeError(f"halo_dilate3x3_f32() fehlgeschlagen (rc={rc}).")

    def open3x3_f32(
        self,
        src: ArrayLikeFloat,
        dst: ArrayLikeFloat,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst_stride_bytes: int,
        *,
        use_mt: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width/height müssen > 0 sein.")
        if src_stride_bytes < width * 4 or dst_stride_bytes < width * 4:
            raise ValueError("Stride zu klein für float32-Bild.")
        c_src = _to_c_f32_ptr_2d(src, width, height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, width, height, dst_stride_bytes)
        rc = _lib.halo_open3x3_f32(
            c_src, C.c_longlong(src_stride_bytes),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(width), C.c_int(height),
            C.c_int(1 if use_mt else 0),
        )
        if rc != 0:
            raise RuntimeError(f"halo_open3x3_f32() fehlgeschlagen (rc={rc}).")

    def close3x3_f32(
        self,
        src: ArrayLikeFloat,
        dst: ArrayLikeFloat,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst_stride_bytes: int,
        *,
        use_mt: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width/height müssen > 0 sein.")
        if src_stride_bytes < width * 4 or dst_stride_bytes < width * 4:
            raise ValueError("Stride zu klein für float32-Bild.")
        c_src = _to_c_f32_ptr_2d(src, width, height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, width, height, dst_stride_bytes)
        rc = _lib.halo_close3x3_f32(
            c_src, C.c_longlong(src_stride_bytes),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(width), C.c_int(height),
            C.c_int(1 if use_mt else 0),
        )
        if rc != 0:
            raise RuntimeError(f"halo_close3x3_f32() fehlgeschlagen (rc={rc}).")

    def unsharp_mask_f32(
        self,
        src: ArrayLikeFloat,
        dst: ArrayLikeFloat,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst_stride_bytes: int,
        *,
        sigma: float,
        amount: float = 1.0,
        threshold: float = 0.0,
        use_mt: bool = True,
    ) -> None:
        if sigma < 0.0:
            raise ValueError("sigma muss >= 0 sein.")
        if width <= 0 or height <= 0:
            raise ValueError("width/height müssen > 0 sein.")
        if src_stride_bytes < width * 4 or dst_stride_bytes < width * 4:
            raise ValueError("Stride zu klein für float32-Bild.")
        c_src = _to_c_f32_ptr_2d(src, width, height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, width, height, dst_stride_bytes)
        rc = _lib.halo_unsharp_mask_f32(
            c_src, C.c_longlong(src_stride_bytes),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(width), C.c_int(height),
            C.c_float(sigma),
            C.c_float(amount),
            C.c_float(threshold),
            C.c_int(1 if use_mt else 0),
        )
        if rc != 0:
            raise RuntimeError(f"halo_unsharp_mask_f32() fehlgeschlagen (rc={rc}).")

    def render_vector_scene(
        self,
        shapes: Sequence[VectorShape] | VectorShape,
        dst: ArrayLikeFloat,
        width: int,
        height: int,
        dst_stride_bytes: int,
        *,
        background: float = 0.0,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width/height müssen > 0 sein.")
        if dst_stride_bytes < width * 4:
            raise ValueError("dst_stride_bytes zu klein für Canvas-Breite")
        shape_list = [shapes] if isinstance(shapes, VectorShape) else list(shapes)
        canvas = VectorCanvas(width, height, background=background)
        for shape in shape_list:
            canvas.draw_shape(shape)
        canvas.blit_to(dst, dst_stride_bytes)

    def render_line_plot(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
        dst: ArrayLikeFloat,
        width: int,
        height: int,
        dst_stride_bytes: int,
        *,
        background: float = 0.0,
        color: float = 1.0,
        tick_count: int = 5,
        x_label: str = "",
        y_label: str = "",
        title: str = "",
    ) -> None:
        if dst_stride_bytes < width * 4:
            raise ValueError("dst_stride_bytes zu klein für Canvas-Breite")
        canvas = VectorCanvas(width, height, background=background)
        layout = PlotLayout(canvas)
        layout.plot_line(xs, ys, color, tick_count=tick_count,
                         x_label=x_label, y_label=y_label, title=title)
        canvas.blit_to(dst, dst_stride_bytes)

    def box_blur_f32(
        self,
        src: ArrayLikeFloat,
        dst: ArrayLikeFloat,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst_stride_bytes: int,
        radius: int,
        *,
        use_mt: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width/height müssen > 0 sein.")
        c_src = _to_c_f32_ptr_2d(src, width, height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, width, height, dst_stride_bytes)
        rc = _lib.halo_box_blur_f32(
            c_src, C.c_longlong(src_stride_bytes),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(width), C.c_int(height),
            C.c_int(radius),
            C.c_int(1 if use_mt else 0)
        )
        if rc != 0:
            raise RuntimeError(f"halo_box_blur_f32() fehlgeschlagen (rc={rc}).")

    def gaussian_blur_f32(
        self,
        src: ArrayLikeFloat,
        dst: ArrayLikeFloat,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst_stride_bytes: int,
        sigma: float,
        *,
        use_mt: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width/height müssen > 0 sein.")
        c_src = _to_c_f32_ptr_2d(src, width, height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, width, height, dst_stride_bytes)
        rc = _lib.halo_gaussian_blur_f32(
            c_src, C.c_longlong(src_stride_bytes),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(width), C.c_int(height),
            C.c_float(sigma),
            C.c_int(1 if use_mt else 0)
        )
        if rc != 0:
            raise RuntimeError(f"halo_gaussian_blur_f32() fehlgeschlagen (rc={rc}).")

    def sobel_f32(
        self,
        src: ArrayLikeFloat,
        dst: ArrayLikeFloat,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst_stride_bytes: int,
        *,
        use_mt: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width/height müssen > 0 sein.")
        c_src = _to_c_f32_ptr_2d(src, width, height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, width, height, dst_stride_bytes)
        rc = _lib.halo_sobel_f32(
            c_src, C.c_longlong(src_stride_bytes),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(width), C.c_int(height),
            C.c_int(1 if use_mt else 0)
        )
        if rc != 0:
            raise RuntimeError(f"halo_sobel_f32() fehlgeschlagen (rc={rc}).")

    def resize_bilinear_f32(
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
        use_mt: bool = True,
    ) -> None:
        if src_width <= 0 or src_height <= 0 or dst_width <= 0 or dst_height <= 0:
            raise ValueError("Quell- und Zielgrößen müssen > 0 sein.")
        c_src = _to_c_f32_ptr_2d(src, src_width, src_height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, dst_width, dst_height, dst_stride_bytes)
        rc = _lib.halo_resize_bilinear_f32(
            c_src, C.c_longlong(src_stride_bytes),
            C.c_int(src_width), C.c_int(src_height),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(dst_width), C.c_int(dst_height),
            C.c_int(1 if use_mt else 0)
        )
        if rc != 0:
            raise RuntimeError(f"halo_resize_bilinear_f32() fehlgeschlagen (rc={rc}).")

    def resize_bicubic_f32(
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
        use_mt: bool = True,
    ) -> None:
        if src_width <= 0 or src_height <= 0 or dst_width <= 0 or dst_height <= 0:
            raise ValueError("Quell- und Zielgrößen müssen > 0 sein.")
        c_src = _to_c_f32_ptr_2d(src, src_width, src_height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, dst_width, dst_height, dst_stride_bytes)
        rc = _lib.halo_resize_bicubic_f32(
            c_src, C.c_longlong(src_stride_bytes),
            C.c_int(src_width), C.c_int(src_height),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(dst_width), C.c_int(dst_height),
            C.c_int(1 if use_mt else 0)
        )
        if rc != 0:
            raise RuntimeError(f"halo_resize_bicubic_f32() fehlgeschlagen (rc={rc}).")

    def relu_clamp_axpby_f32(
        self,
        src: ArrayLikeFloat,
        dst: ArrayLikeFloat,
        width: int,
        height: int,
        src_stride_bytes: int,
        dst_stride_bytes: int,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        clamp_min: float = float('-inf'),
        clamp_max: float = float('inf'),
        apply_relu: bool = False,
        use_mt: bool = True,
    ) -> None:
        c_src = _to_c_f32_ptr_2d(src, width, height, src_stride_bytes)
        c_dst = _to_c_f32_ptr_2d(dst, width, height, dst_stride_bytes)
        rc = _lib.halo_relu_clamp_axpby_f32(
            c_src, C.c_longlong(src_stride_bytes),
            c_dst, C.c_longlong(dst_stride_bytes),
            C.c_int(width), C.c_int(height),
            C.c_float(alpha), C.c_float(beta),
            C.c_float(clamp_min), C.c_float(clamp_max),
            C.c_int(1 if apply_relu else 0),
            C.c_int(1 if use_mt else 0)
        )
        if rc != 0:
            raise RuntimeError(f"halo_relu_clamp_axpby_f32() fehlgeschlagen (rc={rc}).")

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
